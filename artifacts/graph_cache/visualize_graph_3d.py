import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import networkx as nx
import plotly.graph_objects as go

from graph.structures import PixelGraph


PT_PATH = PROJECT_ROOT / "artifacts" / "graph_cache" / "train_graphs.pt"
OUT_DIR = PROJECT_ROOT / "artifacts" / "graph_cache" / "graph_viz_3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Load
# =========================
def load_graphs(pt_path=PT_PATH):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    return data


# =========================
# Utils
# =========================
def idx_to_xy(idx, width):
    y = idx // width
    x = idx % width
    return x, y


def xy_to_idx(x, y, width):
    return y * width + x


def build_nx_graph(pixel_graph: PixelGraph):
    G = nx.Graph()

    node_features = pixel_graph.node_features
    edge_index = pixel_graph.edge_index
    edge_attr = pixel_graph.edge_attr

    num_nodes = node_features.shape[0]
    num_edges = edge_index.shape[1]

    # node_features: [intensity, x_norm, y_norm]
    for i in range(num_nodes):
        intensity = float(node_features[i, 0])
        x_norm = float(node_features[i, 1])
        y_norm = float(node_features[i, 2])

        x, y = idx_to_xy(i, pixel_graph.width)

        G.add_node(
            i,
            intensity=intensity,
            x=x,
            y=y,
            x_norm=x_norm,
            y_norm=y_norm,
            feature_vector=node_features[i].tolist(),
        )

    # edge_attr: ['dx', 'dy', 'dist', 'delta_intensity', 'intensity_similarity']
    for e in range(num_edges):
        u = int(edge_index[0, e])
        v = int(edge_index[1, e])

        attrs = {
            "dx": float(edge_attr[e, 0]),
            "dy": float(edge_attr[e, 1]),
            "dist": float(edge_attr[e, 2]),
            "delta_intensity": float(edge_attr[e, 3]),
            "intensity_similarity": float(edge_attr[e, 4]),
            "edge_id": e,
        }
        G.add_edge(u, v, **attrs)

    return G


def extract_window_subgraph(G, x_min, x_max, y_min, y_max):
    nodes = []
    for n, d in G.nodes(data=True):
        x, y = d["x"], d["y"]
        if x_min <= x <= x_max and y_min <= y <= y_max:
            nodes.append(n)
    return G.subgraph(nodes).copy()


def extract_k_hop_subgraph(G, center_node, k=2):
    nodes = nx.single_source_shortest_path_length(G, center_node, cutoff=k).keys()
    return G.subgraph(nodes).copy()


def normalize_intensity(values):
    values = np.asarray(values, dtype=float)
    vmin = values.min() if len(values) else 0.0
    vmax = values.max() if len(values) else 1.0
    if abs(vmax - vmin) < 1e-12:
        return np.zeros_like(values) + 0.5
    return (values - vmin) / (vmax - vmin)


# =========================
# Plot helpers
# =========================
def build_edge_records(G, z_scale=20.0, stride=1):
    """
    Chuẩn bị edge geometry để vừa render vừa dùng cho tương tác click.
    """
    edge_records = []

    edges = list(G.edges(data=True))
    if stride > 1:
        edges = edges[::stride]

    for u, v, d in edges:
        du = G.nodes[u]
        dv = G.nodes[v]

        x0, y0, z0 = du["x"], -du["y"], du["intensity"] * z_scale
        x1, y1, z1 = dv["x"], -dv["y"], dv["intensity"] * z_scale
        edge_records.append(
            {
                "u": u,
                "v": v,
                "edge_id": d.get("edge_id", "?"),
                "x": [x0, x1],
                "y": [y0, y1],
                "z": [z0, z1],
                "mid_x": (x0 + x1) / 2.0,
                "mid_y": (y0 + y1) / 2.0,
                "mid_z": (z0 + z1) / 2.0,
                "hover": (
                    f"edge {d.get('edge_id', '?')}: {u} ↔ {v}<br>"
                    f"dx={d['dx']:.4f}<br>"
                    f"dy={d['dy']:.4f}<br>"
                    f"dist={d['dist']:.4f}<br>"
                    f"delta_intensity={d['delta_intensity']:.4f}<br>"
                    f"intensity_similarity={d['intensity_similarity']:.4f}"
                ),
            }
        )

    return edge_records


def build_edge_picker_trace_3d(edge_records):
    """
    Dùng marker tại trung điểm edge để dễ click hơn trong không gian 3D.
    """
    xs = [r["mid_x"] for r in edge_records]
    ys = [r["mid_y"] for r in edge_records]
    zs = [r["mid_z"] for r in edge_records]
    hover_texts = [r["hover"] for r in edge_records]
    customdata = [[r["u"], r["v"], r["edge_id"], idx] for idx, r in enumerate(edge_records)]

    trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        hovertext=hover_texts,
        hoverinfo="text",
        customdata=customdata,
        marker=dict(
            size=4,
            color="rgba(44, 132, 232, 0.28)",
            line=dict(width=0.5, color="rgba(20, 70, 130, 0.35)"),
        ),
        name="edge-pickers"
    )
    return trace


def build_all_edges_trace_3d(edge_records):
    xs, ys, zs = [], [], []

    for record in edge_records:
        xs.extend([record["x"][0], record["x"][1], None])
        ys.extend([record["y"][0], record["y"][1], None])
        zs.extend([record["z"][0], record["z"][1], None])

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        hoverinfo="none",
        line=dict(width=1.5, color="rgba(90, 90, 90, 0.18)"),
        visible=False,
        name="all-edges"
    )


def build_highlight_edge_trace_3d():
    return go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode="lines",
        hoverinfo="none",
        line=dict(width=7, color="rgba(255, 127, 14, 0.95)"),
        name="selected-edges"
    )


def build_node_trace_3d(G, z_scale=20.0, size_scale=8.0, show_text=False):
    node_ids = list(G.nodes())
    intensities = np.array([G.nodes[n]["intensity"] for n in node_ids], dtype=float)
    norm_int = normalize_intensity(intensities)

    xs = []
    ys = []
    zs = []
    sizes = []
    texts = []
    labels = []

    for i, n in enumerate(node_ids):
        d = G.nodes[n]
        x = d["x"]
        y = -d["y"]
        z = d["intensity"] * z_scale

        xs.append(x)
        ys.append(y)
        zs.append(z)

        # node size phụ thuộc intensity
        sizes.append(5 + norm_int[i] * size_scale)

        fv = d["feature_vector"]
        texts.append(
            f"<b>node {n}</b><br>"
            f"x={d['x']}, y={d['y']}<br>"
            f"intensity={d['intensity']:.4f}<br>"
            f"x_norm={d['x_norm']:.4f}<br>"
            f"y_norm={d['y_norm']:.4f}<br>"
            f"feature={np.array2string(np.array(fv), precision=4, suppress_small=True)}"
        )
        labels.append(str(n) if show_text else "")

    trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers+text" if show_text else "markers",
        text=labels,
        textposition="top center",
        hovertext=texts,
        hoverinfo="text",
        customdata=node_ids,
        marker=dict(
            size=sizes,
            color=intensities,
            colorscale="Gray",
            colorbar=dict(title="Intensity"),
            opacity=0.95,
            line=dict(width=0.5, color="black"),
        ),
        name="nodes"
    )
    return trace


def build_selected_node_trace_3d(show_text=False):
    return go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode="markers+text" if show_text else "markers",
        text=[],
        textposition="top center",
        hoverinfo="skip",
        marker=dict(
            size=11,
            color="rgba(220, 53, 69, 0.98)",
            opacity=1.0,
            line=dict(width=2, color="white"),
            symbol="circle",
        ),
        name="selected-nodes"
    )


def build_vertical_drop_lines(G, z_scale=20.0):
    """
    Vẽ line từ node xuống mặt phẳng z=0 để dễ đọc không gian 3D.
    """
    xs, ys, zs = [], [], []

    for n, d in G.nodes(data=True):
        x = d["x"]
        y = -d["y"]
        z = d["intensity"] * z_scale

        xs.extend([x, x, None])
        ys.extend([y, y, None])
        zs.extend([0, z, None])

    trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(width=1, color="rgba(80,80,200,0.15)"),
        hoverinfo="none",
        name="drop-lines"
    )
    return trace


def build_base_plane(width, height):
    x = np.array([[0, width - 1], [0, width - 1]])
    y = np.array([[0, 0], [-(height - 1), -(height - 1)]])
    z = np.zeros((2, 2))

    plane = go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.10,
        showscale=False,
        colorscale=[[0, "#d0e3ff"], [1, "#d0e3ff"]],
        hoverinfo="skip",
        name="base-plane"
    )
    return plane


def save_3d_html(
    G,
    save_path,
    title,
    width,
    height,
    z_scale=20.0,
    edge_stride=1,
    show_text=False,
    show_drop_lines=True,
):
    edge_records = build_edge_records(G, z_scale=z_scale, stride=edge_stride)
    enable_edge_pickers = len(edge_records) <= 1500
    node_to_edges = {}
    for idx, record in enumerate(edge_records):
        node_to_edges.setdefault(record["u"], []).append(idx)
        node_to_edges.setdefault(record["v"], []).append(idx)

    traces = []
    trace_index_map = {}

    trace_index_map["base_plane"] = len(traces)
    traces.append(build_base_plane(width, height))
    if show_drop_lines:
        trace_index_map["drop_lines"] = len(traces)
        traces.append(build_vertical_drop_lines(G, z_scale=z_scale))
    trace_index_map["all_edges"] = len(traces)
    traces.append(build_all_edges_trace_3d(edge_records))
    trace_index_map["edge_picker"] = len(traces)
    edge_picker_trace = build_edge_picker_trace_3d(edge_records)
    if not enable_edge_pickers:
        edge_picker_trace.visible = False
    traces.append(edge_picker_trace)
    trace_index_map["selected_edges"] = len(traces)
    traces.append(build_highlight_edge_trace_3d())
    trace_index_map["nodes"] = len(traces)
    traces.append(build_node_trace_3d(G, z_scale=z_scale, show_text=show_text))
    trace_index_map["selected_nodes"] = len(traces)
    traces.append(build_selected_node_trace_3d(show_text=show_text))

    fig = go.Figure(data=traces)

    all_edges_visibility = [False] * len(traces)
    all_edges_visibility[trace_index_map["all_edges"]] = True
    if "drop_lines" in trace_index_map:
        drop_lines_visibility = [False] * len(traces)
        drop_lines_visibility[trace_index_map["drop_lines"]] = True
    else:
        drop_lines_visibility = [False] * len(traces)
    edge_picker_visibility = [False] * len(traces)
    edge_picker_visibility[trace_index_map["edge_picker"]] = True

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (pixel)",
            yaxis_title="Y (pixel)",
            zaxis_title="Intensity height",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.9)
            ),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_white",
        height=900,
        annotations=[
            dict(
                text="Click node to show incident edges. Click blue edge midpoint to show its two endpoints. Double-click to reset. Use the buttons to toggle layers.",
                x=0.5,
                y=1.0,
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=12, color="#3b3b3b"),
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.01,
                y=1.08,
                xanchor="left",
                yanchor="top",
                showactive=True,
                buttons=[
                    dict(
                        label="All edges on",
                        method="restyle",
                        args=[{"visible": True}, [trace_index_map["all_edges"]]],
                    ),
                    dict(
                        label="All edges off",
                        method="restyle",
                        args=[{"visible": False}, [trace_index_map["all_edges"]]],
                    ),
                    dict(
                        label="Edge pickers on",
                        method="restyle",
                        args=[{"visible": True}, [trace_index_map["edge_picker"]]],
                    ),
                    dict(
                        label="Edge pickers off",
                        method="restyle",
                        args=[{"visible": False}, [trace_index_map["edge_picker"]]],
                    ),
                    dict(
                        label="Drop lines on",
                        method="restyle",
                        args=[{"visible": True}, [trace_index_map["drop_lines"]]] if "drop_lines" in trace_index_map else [{}, []],
                    ),
                    dict(
                        label="Drop lines off",
                        method="restyle",
                        args=[{"visible": False}, [trace_index_map["drop_lines"]]] if "drop_lines" in trace_index_map else [{}, []],
                    ),
                ],
            )
        ],
        meta=dict(
            edgeRecords=edge_records,
            nodeToEdges=node_to_edges,
            traceIndexMap=trace_index_map,
            enableEdgePickers=enable_edge_pickers,
        ),
    )

    post_script = """
const gd = document.getElementById('{plot_id}');
const meta = gd.layout.meta || {};
const traceIndexMap = meta.traceIndexMap || {};
const edgeRecords = meta.edgeRecords || [];
const nodeToEdges = meta.nodeToEdges || {};
const enableEdgePickers = !!meta.enableEdgePickers;
const nodeTrace = gd.data[traceIndexMap.nodes];
const nodeIndexById = {};

for (let i = 0; i < nodeTrace.customdata.length; i += 1) {
    nodeIndexById[String(nodeTrace.customdata[i])] = i;
}

function getTrace(name) {
    return gd.data[traceIndexMap[name]];
}

function applySelection(edgeX, edgeY, edgeZ, nodeX, nodeY, nodeZ, nodeText) {
    Plotly.restyle(
        gd,
        {
            x: [edgeX, nodeX],
            y: [edgeY, nodeY],
            z: [edgeZ, nodeZ],
            text: [[], nodeText],
        },
        [traceIndexMap.selected_edges, traceIndexMap.selected_nodes]
    );
}

function clearSelection() {
    applySelection([], [], [], [], [], [], []);
}

function setNodeSelection(selectedNodeIds) {
    const selectedX = [];
    const selectedY = [];
    const selectedZ = [];
    const selectedText = [];

    selectedNodeIds.forEach((nodeId) => {
        const idx = nodeIndexById[String(nodeId)];
        if (idx >= 0) {
            selectedX.push(nodeTrace.x[idx]);
            selectedY.push(nodeTrace.y[idx]);
            selectedZ.push(nodeTrace.z[idx]);
            selectedText.push(String(nodeId));
        }
    });
    applySelection([], [], [], selectedX, selectedY, selectedZ, selectedText);
}

function setEdgeSelection(edgeIndices) {
    const xs = [];
    const ys = [];
    const zs = [];
    const selectedNodes = new Set();

    edgeIndices.forEach((edgeIdx) => {
        const edge = edgeRecords[edgeIdx];
        if (!edge) {
            return;
        }
        xs.push(edge.x[0], edge.x[1], null);
        ys.push(edge.y[0], edge.y[1], null);
        zs.push(edge.z[0], edge.z[1], null);
        selectedNodes.add(edge.u);
        selectedNodes.add(edge.v);
    });

    const selectedX = [];
    const selectedY = [];
    const selectedZ = [];
    const selectedText = [];

    Array.from(selectedNodes).forEach((nodeId) => {
        const idx = nodeIndexById[String(nodeId)];
        if (idx >= 0) {
            selectedX.push(nodeTrace.x[idx]);
            selectedY.push(nodeTrace.y[idx]);
            selectedZ.push(nodeTrace.z[idx]);
            selectedText.push(String(nodeId));
        }
    });

    applySelection(xs, ys, zs, selectedX, selectedY, selectedZ, selectedText);
}

clearSelection();

gd.on('plotly_click', (eventData) => {
    if (!eventData || !eventData.points || !eventData.points.length) {
        return;
    }

    const point = eventData.points[0];
    if (point.curveNumber === traceIndexMap.nodes) {
        const nodeId = point.customdata;
        const edgeIndices = nodeToEdges[String(nodeId)] || nodeToEdges[nodeId] || [];
        if (edgeIndices.length) {
            setEdgeSelection(edgeIndices);
        } else {
            setNodeSelection([nodeId]);
            Plotly.restyle(
                gd,
                { x: [[]], y: [[]], z: [[]] },
                [traceIndexMap.selected_edges]
            );
        }
    } else if (enableEdgePickers && point.curveNumber === traceIndexMap.edge_picker) {
        const edgeIdx = point.customdata[3];
        setEdgeSelection([edgeIdx]);
    }
});

gd.on('plotly_doubleclick', () => {
    clearSelection();
    return false;
});
"""

    fig.write_html(
        str(save_path),
        include_plotlyjs=True,
        post_script=post_script,
    )
    print(f"Saved 3D interactive graph to: {save_path}")


# =========================
# Additional exports
# =========================
def save_feature_panels(g: PixelGraph, save_path):
    """
    Export 2D heatmap summary để nhìn nhanh feature map.
    """
    import plotly.subplots as sp

    intensity = g.node_features[:, 0].reshape(g.height, g.width)
    x_norm = g.node_features[:, 1].reshape(g.height, g.width)
    y_norm = g.node_features[:, 2].reshape(g.height, g.width)

    fig = sp.make_subplots(
        rows=1, cols=3,
        subplot_titles=["Intensity", "x_norm", "y_norm"]
    )

    fig.add_trace(go.Heatmap(z=intensity, colorscale="Gray", showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=x_norm, colorscale="Viridis", showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=y_norm, colorscale="Plasma", showscale=False), row=1, col=3)

    fig.update_layout(
        title="Node feature maps",
        height=450,
        template="plotly_white"
    )
    fig.write_html(str(save_path))
    print(f"Saved feature panel to: {save_path}")


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_id", type=int, default=0)

    parser.add_argument(
        "--mode",
        type=str,
        default="window",
        choices=["full", "window", "khop"]
    )

    parser.add_argument("--x_min", type=int, default=18)
    parser.add_argument("--x_max", type=int, default=29)
    parser.add_argument("--y_min", type=int, default=18)
    parser.add_argument("--y_max", type=int, default=29)

    parser.add_argument("--center_x", type=int, default=24)
    parser.add_argument("--center_y", type=int, default=24)
    parser.add_argument("--k", type=int, default=2)

    parser.add_argument("--z_scale", type=float, default=20.0)
    parser.add_argument("--edge_stride", type=int, default=1)
    parser.add_argument("--show_text", action="store_true")
    parser.add_argument("--no_drop_lines", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    graphs = load_graphs()
    print(f"Loaded {len(graphs)} graphs")

    g = graphs[args.graph_id]
    G = build_nx_graph(g)

    save_feature_panels(g, OUT_DIR / f"graph_{args.graph_id}_feature_maps.html")

    if args.mode == "full":
        SG = G
        title = f"Graph {args.graph_id} - Full 3D Graph"
        save_name = OUT_DIR / f"graph_{args.graph_id}_full_3d.html"

    elif args.mode == "window":
        SG = extract_window_subgraph(
            G,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
        )
        title = (
            f"Graph {args.graph_id} - Window 3D Graph "
            f"(x={args.x_min}..{args.x_max}, y={args.y_min}..{args.y_max})"
        )
        save_name = OUT_DIR / (
            f"graph_{args.graph_id}_window_"
            f"x{args.x_min}-{args.x_max}_y{args.y_min}-{args.y_max}_3d.html"
        )

    else:
        center_node = xy_to_idx(args.center_x, args.center_y, g.width)
        SG = extract_k_hop_subgraph(G, center_node=center_node, k=args.k)
        title = (
            f"Graph {args.graph_id} - {args.k}-hop 3D Graph "
            f"around node {center_node} ({args.center_x}, {args.center_y})"
        )
        save_name = OUT_DIR / (
            f"graph_{args.graph_id}_khop_center_{args.center_x}_{args.center_y}_k{args.k}_3d.html"
        )

    save_3d_html(
        SG,
        save_path=save_name,
        title=title,
        width=g.width,
        height=g.height,
        z_scale=args.z_scale,
        edge_stride=args.edge_stride,
        show_text=args.show_text,
        show_drop_lines=not args.no_drop_lines,
    )

    print("Done.")


if __name__ == "__main__":
    main()
