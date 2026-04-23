import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from graph.structures import PixelGraph

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False


PT_PATH = PROJECT_ROOT / "artifacts" / "graph_cache" / "train_graphs.pt"
OUT_DIR = PROJECT_ROOT / "artifacts" / "graph_cache" / "graph_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_graphs(pt_path=PT_PATH):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    return data


def idx_to_xy(idx, width):
    y = idx // width
    x = idx % width
    return x, y


def build_nx_graph(pixel_graph: PixelGraph):
    G = nx.Graph()

    num_nodes = pixel_graph.node_features.shape[0]
    node_features = pixel_graph.node_features
    edge_index = pixel_graph.edge_index
    edge_attr = pixel_graph.edge_attr

    # Node features:
    # [intensity, x_norm, y_norm]
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
            y_norm=y_norm
        )

    # Edge features:
    # ['dx', 'dy', 'dist', 'delta_intensity', 'intensity_similarity']
    num_edges = edge_index.shape[1]
    for e in range(num_edges):
        u = int(edge_index[0, e])
        v = int(edge_index[1, e])

        attrs = {}
        if edge_attr is not None:
            attrs = {
                "dx": float(edge_attr[e, 0]),
                "dy": float(edge_attr[e, 1]),
                "dist": float(edge_attr[e, 2]),
                "delta_intensity": float(edge_attr[e, 3]),
                "intensity_similarity": float(edge_attr[e, 4]),
            }

        G.add_edge(u, v, **attrs)

    return G


def get_pixel_layout(G):
    """
    Layout theo đúng vị trí pixel.
    Dùng để xem graph bám theo ảnh 48x48.
    """
    pos = {}
    for n, d in G.nodes(data=True):
        # đảo trục y để nhìn giống ảnh
        pos[n] = (d["x"], -d["y"])
    return pos


def draw_full_graph(G, save_path):
    """
    Vẽ toàn bộ graph.
    Lưu ý: nhiều node + edge nên khá rối, nhưng cho cái nhìn tổng thể.
    """
    pos = get_pixel_layout(G)

    intensities = [G.nodes[n]["intensity"] for n in G.nodes()]

    plt.figure(figsize=(12, 12))
    nx.draw_networkx_edges(
        G, pos,
        width=0.2,
        alpha=0.15
    )
    nx.draw_networkx_nodes(
        G, pos,
        node_size=8,
        node_color=intensities,
        cmap="gray"
    )

    plt.title("Full Pixel Graph (layout theo ảnh)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved full graph to: {save_path}")


def extract_window_subgraph(G, width, x_min, x_max, y_min, y_max):
    """
    Cắt subgraph theo cửa sổ pixel.
    """
    selected_nodes = []

    for n, d in G.nodes(data=True):
        x, y = d["x"], d["y"]
        if x_min <= x <= x_max and y_min <= y <= y_max:
            selected_nodes.append(n)

    SG = G.subgraph(selected_nodes).copy()
    return SG


def extract_k_hop_subgraph(G, center_node, k=1):
    """
    Lấy k-hop subgraph quanh 1 node.
    """
    nodes = nx.single_source_shortest_path_length(G, center_node, cutoff=k).keys()
    SG = G.subgraph(nodes).copy()
    return SG


def draw_subgraph(G, save_path, title="Subgraph", node_size=120):
    pos = get_pixel_layout(G)
    intensities = [G.nodes[n]["intensity"] for n in G.nodes()]

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_edges(
        G, pos,
        width=0.8,
        alpha=0.5
    )
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color=intensities,
        cmap="gray"
    )
    nx.draw_networkx_labels(
        G, pos,
        font_size=7
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved subgraph to: {save_path}")


def export_pyvis_html(G, save_path, title="Interactive Graph"):
    if not HAS_PYVIS:
        print("PyVis chưa được cài. Hãy chạy: pip install pyvis")
        return

    net = Network(height="900px", width="100%", bgcolor="white", font_color="black")
    net.barnes_hut()

    for n, d in G.nodes(data=True):
        intensity = d.get("intensity", 0.0)
        label = str(n)

        tooltip = (
            f"node={n}<br>"
            f"x={d.get('x')} y={d.get('y')}<br>"
            f"intensity={intensity:.4f}"
        )

        # intensity -> grayscale
        gray = int(max(0, min(255, intensity * 255)))
        color = f"rgb({gray},{gray},{gray})"

        net.add_node(
            n,
            label=label,
            title=tooltip,
            color=color
        )

    for u, v, d in G.edges(data=True):
        sim = d.get("intensity_similarity", None)
        dist = d.get("dist", None)

        tooltip = (
            f"{u} -> {v}<br>"
            f"dist={dist}<br>"
            f"sim={sim}"
        )

        net.add_edge(
            u, v,
            title=tooltip
        )

    net.write_html(str(save_path))
    print(f"Saved interactive html to: {save_path}")


def main():
    graphs = load_graphs()
    print(f"Loaded {len(graphs)} graphs")

    g0 = graphs[0]
    print("Graph 0:", g0)

    G = build_nx_graph(g0)

    # 1) Full graph
    draw_full_graph(
        G,
        OUT_DIR / "graph0_full.png"
    )

    # 2) Window subgraph: ví dụ vùng giữa ảnh
    # 48x48 -> lấy vùng 18..29 cả x,y (12x12)
    SG_window = extract_window_subgraph(
        G,
        width=g0.width,
        x_min=18, x_max=29,
        y_min=18, y_max=29
    )
    draw_subgraph(
        SG_window,
        OUT_DIR / "graph0_window_12x12.png",
        title="Window Subgraph (x=18..29, y=18..29)",
        node_size=150
    )
    export_pyvis_html(
        SG_window,
        OUT_DIR / "graph0_window_12x12.html",
        title="Window Subgraph"
    )

    # 3) k-hop subgraph quanh node trung tâm
    center_x, center_y = 24, 24
    center_node = center_y * g0.width + center_x
    SG_khop = extract_k_hop_subgraph(G, center_node=center_node, k=2)

    draw_subgraph(
        SG_khop,
        OUT_DIR / "graph0_center_k2.png",
        title=f"2-hop Subgraph around node {center_node} ({center_x},{center_y})",
        node_size=250
    )
    export_pyvis_html(
        SG_khop,
        OUT_DIR / "graph0_center_k2.html",
        title="2-hop Subgraph"
    )

    print("Done.")


if __name__ == "__main__":
    main()