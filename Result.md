
tổng hợp kết quả vào file md cho tôi
đây là ce_only
===============================
1934.7s	5297	TEST SET EVALUATION
1934.7s	5298	=======================================================
1934.7s	5299	Accuracy:    33.44%
1934.7s	5300	Macro F1:    0.2139
1934.7s	5301	Weighted F1: 0.2688
1934.7s	5302	pred_count:  [2, 0, 94, 2248, 531, 191, 523]
1934.7s	5303	
1934.7s	5304	Classification report:
1934.7s	5305	Angry          precision=0.5000 recall=0.0020 f1=0.0041 support=491
1934.7s	5306	Disgust        precision=0.0000 recall=0.0000 f1=0.0000 support=55
1934.7s	5307	Fear           precision=0.2979 recall=0.0530 f1=0.0900 support=528
1934.7s	5308	Happy          precision=0.3243 recall=0.8294 f1=0.4663 support=879
1934.7s	5309	Sad            precision=0.2994 recall=0.2677 f1=0.2827 support=594
1934.7s	5310	Surprise       precision=0.5445 recall=0.2500 f1=0.3427 support=416
1934.7s	5311	Neutral        precision=0.3423 recall=0.2859 f1=0.3116 support=626
1934.7s	5312	accuracy       0.3344
1934.7s	5313	macro avg      precision=0.3298 recall=0.2411 f1=0.2139 support=3589
1934.7s	5314	weighted avg   precision=0.3640 recall=0.3344 f1=0.2688 support=3589
1934.7s	5315	Confusion matrix saved: /kaggle/working/outputs/d5a_ce_only/20260429_124802/evaluation/confusion_matrix.png
1934.7s	5316	Classification report saved: /kaggle/working/outputs/d5a_ce_only/20260429_124802/evaluation/classification_report.txt
1934.7s	5317	Correct examples saved: /kaggle/working/outputs/d5a_ce_only/20260429_124802/evaluation/correct_examples.png
1934.7s	5318	Wrong examples saved: /kaggle/working/outputs/d5a_ce_only/20260429_124802/evaluation/wrong_examples.png
1934.7s	5319	Evaluation outputs: /kaggle/working/outputs/d5a_ce_only/20260429_124802/evaluation
1935.8s	5320	Visualize command: /usr/bin/python3 scripts/run_experiment.py --config configs/experiments/d5a_ce_only.yaml --environment kaggle --mode visualize --checkpoint /kaggle/working/outputs/d5a_ce_only/20260429_124802/checkpoints/best.pth --graph_repo_path /kaggle/input/datasets/irthn1311/graph-repo/graph_repo --device cuda:0 --chunk_cache_size 8
1935.8s	5321	================================================================================
1956.9s	5322	resolved run.mode: visualize
1956.9s	5323	resolved run.zip_outputs: False
1956.9s	5324	--- torch version: 2.10.0+cu128
1956.9s	5325	--- cuda available: True
1956.9s	5326	--- cuda device count: 2
1956.9s	5327	--- selected device: cuda:0
1956.9s	5328	--- gpu name: Tesla T4
1956.9s	5329	--- current cuda device: 0
1956.9s	5330	[FullGraphDataset test]
1956.9s	5331	chunk_cache_size=8
1956.9s	5332	num_chunks=8
1956.9s	5333	num_samples=3589
1956.9s	5334	[FullGraphDataset test] chunk cache enabled (max_chunks=8)
1956.9s	5335	[DataLoader split=test] batch_size=32 num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=None chunk_aware_shuffle=False
1956.9s	5336	Class gates: /kaggle/working/outputs/d5a_ce_only/20260429_124802/figures/d5a_class_gates
1956.9s	5337	Attention maps: /kaggle/working/outputs/d5a_ce_only/20260429_124802/figures/d5a_attention
1957.3s	5338	resolved run.mode: visualize
1957.3s	5339	resolved run.zip_outputs: False
1957.3s	5340	--- torch version: 2.10.0+cu128
1957.3s	5341	--- cuda available: True
1957.3s	5342	--- cuda device count: 2
1957.3s	5343	--- selected device: cuda:0
1957.3s	5344	--- gpu name: Tesla T4
1957.3s	5345	--- current cuda device: 0
1957.3s	5346	[FullGraphDataset test]
1957.3s	5347	chunk_cache_size=8
1957.3s	5348	num_chunks=8
1957.3s	5349	num_samples=3589
1957.3s	5350	[FullGraphDataset test] chunk cache enabled (max_chunks=8)
1957.3s	5351	[DataLoader split=test] batch_size=32 num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=None chunk_aware_shuffle=False
1957.3s	5352	Class gates: /kaggle/working/outputs/d5a_ce_only/20260429_124802/figures/d5a_class_gates
1957.3s	5353	Attention maps: /kaggle/working/outputs/d5a_ce_only/20260429_124802/figures/d5a_attention
1958.4s	5354	============================================================
1958.4s	5355	GRAPH REPO:
1958.4s	5356	  /kaggle/working/artifacts/graph_repo/manifest.pt: False
1958.4s	5357	  /kaggle/working/artifacts/graph_repo/shared/shared_graph.pt: False
1958.4s	5358	  /kaggle/working/artifacts/graph_repo/train: False
1958.4s	5359	  /kaggle/working/artifacts/graph_repo/val: False
1958.4s	5360	  /kaggle/working/artifacts/graph_repo/test: False
1958.4s	5361	
1958.4s	5362	============================================================
1958.4s	5363	ZIP FILES:
1958.8s	5364	OUTPUT_DIR: /kaggle/working/outputs/d5a_ce_only/20260429_124802
1958.8s	5365	Checkpoints:
1958.8s	5366	  best.pth  (105.43 MB)
1958.8s	5367	  last.pth  (105.43 MB)
1958.8s	5368	
1958.8s	5369	Metrics:
1958.8s	5370	{
1958.8s	5371	  "accuracy": 0.33435497353023125,
1958.8s	5372	  "macro_f1": 0.2138944792623853,
1958.8s	5373	  "weighted_f1": 0.2688417882318637,
1958.8s	5374	  "classification_report": {
1958.8s	5375	    "Angry": {
1958.8s	5376	      "precision": 0.5,
1958.8s	5377	      "recall": 0.002036659877800407,
1958.8s	5378	      "f1-score": 0.004056795131845842,
1958.8s	5379	      "support": 491.0
1958.8s	5380	    },
1958.8s	5381	    "Disgust": {
1958.8s	5382	      "precision": 0.0,
1958.8s	5383	      "recall": 0.0,
1958.8s	5384	      "f1-score": 0.0,
1958.8s	5385	      "support": 55.0
1958.8s	5386	    },
1958.8s	5387	    "Fear": {
1958.8s	5388	      "precision": 0.2978723404255319,
1958.8s	5389	      "recall": 0.05303030303030303,
1958.8s	5390	      "f1-score": 0.09003215434083602,
1958.8s	5391	      "support": 528.0
1958.8s	5392	    },
1958.8s	5393	    "Happy": {
1958.8s	5394	      "precision": 0.32428825622775803,
1958.8s	5395	      "recall": 0.8293515358361775,
1958.8s	5396	      "f1-score": 0.46626159258074834,
1958.8s	5397	      "support": 879.0
1958.8s	5398	    },
1958.8s	5399	    "Sad": {
1958.8s	5400	      "precision": 0.2994350282485876,
1958.8s	5401	      "recall": 0.2676767676767677,
1958.8s	5402	      "f1-score": 0.2826666666666667,
1958.8s	5403	      "support": 594.0
1958.8s	5404	    },
1958.8s	5405	    "Surprise": {
1958.8s	5406	      "precision": 0.5445026178010471,
1958.8s	5407	      "recall": 0.25,
1958.8s	5408	      "f1-score": 0.342668863261944,
1958.8s	5409	      "support": 416.0
1958.8s	5410	    },
1958.8s	5411	    "Neutral": {
1958.8s	5412	      "precision": 0.3422562141491396,
1958.8s	5413	      "recall": 0.28594249201277955,
1958.8s	5414	      "f1-score": 0.31157528285465624,
1958.8s	5415	      "support": 626.0
1958.8s	5416	    },
1958.8s	5417	    "accuracy": 0.33435497353023125,
1958.8s	5418	    "macro avg": {
1958.8s	5419	      "precision": 0.32976492240743777,
1958.8s	5420	      "recall": 0.2411482512048326,
1958.8s	5421	      "f1-score": 0.2138944792623853,
1958.8s	5422	      "support": 3589.0
1958.8s	5423	    },
1958.8s	5424	    "weighted avg": {
1958.8s	5425	      "precision": 0.36401667840934476,
1958.8s	5426	      "recall": 0.33435497353023125,
1958.8s	5427	      "f1-score": 0.2688417882318637,
1958.8s	5428	      "support": 3589.0
1958.8s	5429	    }
1958.8s	5430	  }
1958.8s	5431	}
1958.8s	5432	
1958.8s	5433	/kaggle/working/outputs/d5a_ce_only/20260429_124802/evaluation/confusion_matrix.png
1958.9s	5434	
1958.9s	5435	Figures: 55
1958.9s	5436	  figures/d5a_attention/all_class_grids/sample_0_all_classes.png
1958.9s	5437	  figures/d5a_attention/all_class_grids/sample_10_all_classes.png
1958.9s	5438	  figures/d5a_attention/all_class_grids/sample_11_all_classes.png
1958.9s	5439	  figures/d5a_attention/all_class_grids/sample_12_all_classes.png
1958.9s	5440	  figures/d5a_attention/all_class_grids/sample_13_all_classes.png
1958.9s	5441	  figures/d5a_attention/all_class_grids/sample_14_all_classes.png
1958.9s	5442	  figures/d5a_attention/all_class_grids/sample_15_all_classes.png
1958.9s	5443	  figures/d5a_attention/all_class_grids/sample_1_all_classes.png
1958.9s	5444	  figures/d5a_attention/all_class_grids/sample_2_all_classes.png
1958.9s	5445	  figures/d5a_attention/all_class_grids/sample_3_all_classes.png
1958.9s	5446	  figures/d5a_attention/all_class_grids/sample_4_all_classes.png
1958.9s	5447	  figures/d5a_attention/all_class_grids/sample_5_all_classes.png
1958.9s	5448	  figures/d5a_attention/all_class_grids/sample_6_all_classes.png
1958.9s	5449	  figures/d5a_attention/all_class_grids/sample_7_all_classes.png
1958.9s	5450	  figures/d5a_attention/all_class_grids/sample_8_all_classes.png
1958.9s	5451	  figures/d5a_attention/all_class_grids/sample_9_all_classes.png
1958.9s	5452	  figures/d5a_attention/top_edges/sample_0_pred_edges.png
1958.9s	5453	  figures/d5a_attention/top_edges/sample_10_pred_edges.png
1958.9s	5454	  figures/d5a_attention/top_edges/sample_11_pred_edges.png
1958.9s	5455	  figures/d5a_attention/top_edges/sample_12_pred_edges.png
đây là ce contrast_light
=======================================================
1923.1s	5287	TEST SET EVALUATION
1923.1s	5288	=======================================================
1923.1s	5289	Accuracy:    37.00%
1923.1s	5290	Macro F1:    0.2561
1923.1s	5291	Weighted F1: 0.3155
1923.1s	5292	pred_count:  [26, 0, 131, 1820, 626, 312, 674]
1923.1s	5293	
1923.1s	5294	Classification report:
1923.1s	5295	Angry          precision=0.5385 recall=0.0285 f1=0.0542 support=491
1923.1s	5296	Disgust        precision=0.0000 recall=0.0000 f1=0.0000 support=55
1923.1s	5297	Fear           precision=0.3206 recall=0.0795 f1=0.1275 support=528
1923.1s	5298	Happy          precision=0.3824 recall=0.7918 f1=0.5157 support=879
1923.1s	5299	Sad            precision=0.3019 recall=0.3182 f1=0.3098 support=594
1923.1s	5300	Surprise       precision=0.5032 recall=0.3774 f1=0.4313 support=416
1923.1s	5301	Neutral        precision=0.3412 recall=0.3674 f1=0.3538 support=626
1923.1s	5302	accuracy       0.3700
1923.1s	5303	macro avg      precision=0.3411 recall=0.2804 f1=0.2561 support=3589
1923.1s	5304	weighted avg   precision=0.3823 recall=0.3700 f1=0.3155 support=3589
1923.1s	5305	Confusion matrix saved: /kaggle/working/outputs/d5a_ce_contrast_light/20260429_124802/evaluation/confusion_matrix.png
1923.1s	5306	Classification report saved: /kaggle/working/outputs/d5a_ce_contrast_light/20260429_124802/evaluation/classification_report.txt
1923.1s	5307	Correct examples saved: /kaggle/working/outputs/d5a_ce_contrast_light/20260429_124802/evaluation/correct_examples.png
1923.1s	5308	Wrong examples saved: /kaggle/working/outputs/d5a_ce_contrast_light/20260429_124802/evaluation/wrong_examples.png
1923.1s	5309	Evaluation outputs: /kaggle/working/outputs/d5a_ce_contrast_light/20260429_124802/evaluation
1924.0s	5310	Visualize command: /usr/bin/python3 scripts/run_experiment.py --config configs/experiments/d5a_ce_contrast_light.yaml --environment kaggle --mode visualize --checkpoint /kaggle/working/outputs/d5a_ce_contrast_light/20260429_124802/checkpoints/best.pth --graph_repo_path /kaggle/input/datasets/irthn1311/graph-repo/graph_repo --device cuda:0 --chunk_cache_size 8
1924.0s	5311	================================================================================
1945.1s	5312	resolved run.mode: visualize
1945.1s	5313	resolved run.zip_outputs: False
1945.1s	5314	--- torch version: 2.10.0+cu128
1945.1s	5315	--- cuda available: True
1945.1s	5316	--- cuda device count: 2
1945.1s	5317	--- selected device: cuda:0
1945.1s	5318	--- gpu name: Tesla T4
1945.1s	5319	--- current cuda device: 0
1945.1s	5320	[FullGraphDataset test]
1945.1s	5321	chunk_cache_size=8
1945.1s	5322	num_chunks=8
1945.1s	5323	num_samples=3589
1945.1s	5324	[FullGraphDataset test] chunk cache enabled (max_chunks=8)
1945.1s	5325	[DataLoader split=test] batch_size=32 num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=None chunk_aware_shuffle=False
1945.1s	5326	Class gates: /kaggle/working/outputs/d5a_ce_contrast_light/20260429_124802/figures/d5a_class_gates
1945.1s	5327	Attention maps: /kaggle/working/outputs/d5a_ce_contrast_light/20260429_124802/figures/d5a_attention
1945.3s	5328	resolved run.mode: visualize
1945.3s	5329	resolved run.zip_outputs: False
1945.3s	5330	--- torch version: 2.10.0+cu128
1945.3s	5331	--- cuda available: True
1945.3s	5332	--- cuda device count: 2
1945.3s	5333	--- selected device: cuda:0
1945.3s	5334	--- gpu name: Tesla T4
1945.3s	5335	--- current cuda device: 0
1945.3s	5336	[FullGraphDataset test]
1945.3s	5337	chunk_cache_size=8
1945.3s	5338	num_chunks=8
1945.3s	5339	num_samples=3589
1945.3s	5340	[FullGraphDataset test] chunk cache enabled (max_chunks=8)
1945.3s	5341	[DataLoader split=test] batch_size=32 num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=None chunk_aware_shuffle=False
1945.3s	5342	Class gates: /kaggle/working/outputs/d5a_ce_contrast_light/20260429_124802/figures/d5a_class_gates
1945.3s	5343	Attention maps: /kaggle/working/outputs/d5a_ce_contrast_light/20260429_124802/figures/d5a_attention
1946.5s	5344	============================================================
1946.5s	5345	GRAPH REPO:
1946.5s	5346	  /kaggle/working/artifacts/graph_repo/manifest.pt: False
1946.5s	5347	  /kaggle/working/artifacts/graph_repo/shared/shared_graph.pt: False
1946.5s	5348	  /kaggle/working/artifacts/graph_repo/train: False
1946.5s	5349	  /kaggle/working/artifacts/graph_repo/val: False
1946.5s	5350	  /kaggle/working/artifacts/graph_repo/test: False
1946.5s	5351	
1946.5s	5352	============================================================
1946.5s	5353	ZIP FILES:
1947.0s	5354	OUTPUT_DIR: /kaggle/working/outputs/d5a_ce_contrast_light/20260429_124802
1947.0s	5355	Checkpoints:
1947.0s	5356	  best.pth  (105.43 MB)
1947.0s	5357	  last.pth  (105.43 MB)
1947.0s	5358	
1947.0s	5359	Metrics:
1947.0s	5360	{
1947.0s	5361	  "accuracy": 0.37001950404012257,
1947.0s	5362	  "macro_f1": 0.2560531340361544,
1947.0s	5363	  "weighted_f1": 0.3154678108323314,
1947.0s	5364	  "classification_report": {
1947.0s	5365	    "Angry": {
1947.0s	5366	      "precision": 0.5384615384615384,
1947.0s	5367	      "recall": 0.028513238289205704,
1947.0s	5368	      "f1-score": 0.05415860735009671,
1947.0s	5369	      "support": 491.0
1947.0s	5370	    },
1947.0s	5371	    "Disgust": {
1947.0s	5372	      "precision": 0.0,
1947.0s	5373	      "recall": 0.0,
1947.0s	5374	      "f1-score": 0.0,
1947.0s	5375	      "support": 55.0
1947.0s	5376	    },
1947.0s	5377	    "Fear": {
1947.0s	5378	      "precision": 0.32061068702290074,
1947.0s	5379	      "recall": 0.07954545454545454,
1947.0s	5380	      "f1-score": 0.1274658573596358,
1947.0s	5381	      "support": 528.0
1947.0s	5382	    },
1947.0s	5383	    "Happy": {
1947.0s	5384	      "precision": 0.3824175824175824,
1947.0s	5385	      "recall": 0.7918088737201365,
1947.0s	5386	      "f1-score": 0.5157465728047425,
1947.0s	5387	      "support": 879.0
1947.0s	5388	    },
1947.0s	5389	    "Sad": {
1947.0s	5390	      "precision": 0.3019169329073482,
1947.0s	5391	      "recall": 0.3181818181818182,
1947.0s	5392	      "f1-score": 0.30983606557377047,
1947.0s	5393	      "support": 594.0
1947.0s	5394	    },
1947.0s	5395	    "Surprise": {
1947.0s	5396	      "precision": 0.5032051282051282,
1947.0s	5397	      "recall": 0.37740384615384615,
1947.0s	5398	      "f1-score": 0.43131868131868134,
1947.0s	5399	      "support": 416.0
1947.0s	5400	    },
1947.0s	5401	    "Neutral": {
1947.0s	5402	      "precision": 0.34124629080118696,
1947.0s	5403	      "recall": 0.36741214057507987,
1947.0s	5404	      "f1-score": 0.35384615384615387,
1947.0s	5405	      "support": 626.0
1947.0s	5406	    },
1947.0s	5407	    "accuracy": 0.37001950404012257,
1947.0s	5408	    "macro avg": {
1947.0s	5409	      "precision": 0.34112259425938357,
1947.0s	5410	      "recall": 0.28040933878079155,
1947.0s	5411	      "f1-score": 0.2560531340361544,
1947.0s	5412	      "support": 3589.0
1947.0s	5413	    },
1947.0s	5414	    "weighted avg": {
1947.0s	5415	      "precision": 0.38230824257442275,
1947.0s	5416	      "recall": 0.37001950404012257,
1947.0s	5417	      "f1-score": 0.3154678108323314,
1947.0s	5418	      "support": 3589.0
1947.0s	5419	    }
1947.0s	5420	  }
1947.0s	5421	}
1947.0s	5422	
1947.0s	5423	/kaggle/working/outputs/d5a_ce_contrast_light/20260429_124802/evaluation/confusion_matrix.png
1947.1s	5424	
1947.1s	5425	Figures: 55

