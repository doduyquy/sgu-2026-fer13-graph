d6b_class_part_graph_motif_long100.yaml

=======================================================
7724.8s	357	TEST SET EVALUATION
7724.8s	358	=======================================================
7724.8s	359	Accuracy:    54.64%
7724.8s	360	Macro F1:    0.5181
7724.8s	361	Weighted F1: 0.5422
7724.8s	362	pred_count:  [432, 46, 475, 950, 650, 432, 604]
7724.8s	363	
7724.8s	364	Classification report:
7724.8s	365	Angry          precision=0.4329 recall=0.3809 f1=0.4052 support=491
7724.8s	366	Disgust        precision=0.5217 recall=0.4364 f1=0.4752 support=55
7724.8s	367	Fear           precision=0.3768 recall=0.3390 f1=0.3569 support=528
7724.8s	368	Happy          precision=0.7411 recall=0.8009 f1=0.7698 support=879
7724.8s	369	Sad            precision=0.3954 recall=0.4327 f1=0.4132 support=594
7724.8s	370	Surprise       precision=0.6782 recall=0.7043 f1=0.6910 support=416
7724.8s	371	Neutral        precision=0.5248 recall=0.5064 f1=0.5154 support=626
7724.8s	372	accuracy       0.5464
7724.8s	373	macro avg      precision=0.5244 recall=0.5144 f1=0.5181 support=3589
7724.8s	374	weighted avg   precision=0.5397 recall=0.5464 f1=0.5422 support=3589
7724.8s	375	Confusion matrix saved: /kaggle/working/outputs/d6b_class_part_graph_motif_long100/evaluation/confusion_matrix.png
7724.8s	376	Classification report saved: /kaggle/working/outputs/d6b_class_part_graph_motif_long100/evaluation/classification_report.txt
7724.8s	377	Correct examples saved: /kaggle/working/outputs/d6b_class_part_graph_motif_long100/evaluation/correct_examples.png
7724.8s	378	Wrong examples saved: /kaggle/working/outputs/d6b_class_part_graph_motif_long100/evaluation/wrong_examples.png
7724.8s	379	Evaluation outputs: /kaggle/working/outputs/d6b_class_part_graph_motif_long100/evaluation
7725.9s	380	
7725.9s	381	$ /usr/bin/python3 scripts/visualize_d6.py --config configs/experiments/d6b_class_part_graph_motif_long100.yaml --environment kaggle --checkpoint /kaggle/working/outputs/d6b_class_part_graph_motif_long100/checkpoints/best.pth --graph_repo_path /kaggle/input/datasets/irthn1311/graph-repo/graph_repo --max_samples 32 --device cuda:0 --chunk_cache_size 8
7824.9s	382	--- torch version: 2.10.0+cu128
7824.9s	383	--- cuda available: True
7824.9s	384	--- cuda device count: 2
7824.9s	385	--- selected device: cuda:0
7824.9s	386	--- gpu name: Tesla T4
7824.9s	387	--- current cuda device: 0
7824.9s	388	[FullGraphDataset test]
7824.9s	389	chunk_cache_size=8
7824.9s	390	num_chunks=8
7824.9s	391	num_samples=3589
7824.9s	392	[FullGraphDataset test] chunk cache enabled (max_chunks=8)
7824.9s	393	[DataLoader split=test] batch_size=32 num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=None chunk_aware_shuffle=False
7824.9s	394	D6 part masks: /kaggle/working/outputs/d6b_class_part_graph_motif_long100/figures/d6_part_masks
7824.9s	395	D6 part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_long100/figures/d6_part_attention
7824.9s	396	D6 slot summary: /kaggle/working/outputs/d6b_class_part_graph_motif_long100/figures/d6_slot_summary
7824.9s	397	D6 class-part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_long100/figures/d6_class_part_attention
7824.9s	398	D6 class motif maps: /kaggle/working/outputs/d6b_class_part_graph_motif_long100/figures/d6_class_motif_maps
7826.7s	399	--- torch version: 2.10.0+cu128
7826.7s	400	--- cuda available: True
7826.7s	401	--- cuda device count: 2
7826.7s	402	--- selected device: cuda:0
7826.7s	403	--- gpu name: Tesla T4
7826.7s	404	--- current cuda device: 0
7826.7s	405	[FullGraphDataset test]
7826.7s	406	chunk_cache_size=8
7826.7s	407	num_chunks=8
7826.7s	408	num_samples=3589
7826.7s	409	[FullGraphDataset test] chunk cache enabled (max_chunks=8)
7826.7s	410	[DataLoader split=test] batch_size=32 num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=None chunk_aware_shuffle=False
7826.7s	411	D6 part masks: /kaggle/working/outputs/d6b_class_part_graph_motif_long100/figures/d6_part_masks
7826.7s	412	D6 part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_long100/figures/d6_part_attention
7826.7s	413	D6 slot summary: /kaggle/working/outputs/d6b_class_part_graph_motif_long100/figures/d6_slot_summary
7826.7s	414	D6 class-part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_long100/figures/d6_class_part_attention
7826.7s	415	D6 class motif maps: /kaggle/working/outputs/d6b_class_part_graph_motif_long100/figures/d6_class_motif_maps
7829.7s	416	OUTPUT_DIR: /kaggle/working/outputs/d6b_class_part_graph_motif_long100
7829.7s	417	best epoch: 97
7829.7s	418	best val_macro_f1: 0.5197270096351898
7829.7s	419	lr: 0.00025
7829.7s	420	lr_after_scheduler: 0.00025
7829.7s	421	val_diag_class_part_entropy_mean: 2.674235903056322
7829.7s	422	val_diag_effective_slots: 12.265840361603594
7829.7s	423	
7829.7s	424	TEST
7829.7s	425	accuracy   : 0.5463917525773195
7829.7s	426	macro_f1   : 0.5181235591550932
7829.7s	427	weighted_f1: 0.5421550685127945
7829.7s	428	pred_count : [432, 46, 475, 950, 650, 432, 604]
7829.7s	429	
7829.7s	430	 evaluation/confusion_matrix.png
7829.7s	431	
7829.7s	432	 figures/d6_slot_summary/slot_area.png
7829.7s	433	
7829.7s	434	 figures/d6_slot_summary/slot_similarity.png
7829.7s	435	
7829.7s	436	 figures/d6_class_part_attention/class_part_attn_grid.png
7829.7s	437	
7829.7s	438	 figures/d6_class_part_attention/class_part_attn_avg_by_true_class.png
Class attention CSVs:
7829.7s	441	  figures/d6_class_part_attention/class_part_attention_entropy.csv
7829.7s	442	  figures/d6_class_part_attention/class_part_attention_similarity.csv
7829.7s	443	  figures/d6_class_part_attention/top_slots_per_class.csv
7833.8s	444	ZIP: /kaggle/working/outputs/d6b_class_part_graph_motif_long100.zip



d6b_class_part_graph_motif_long120.yaml

9336.0s	6924	=======================================================
9336.0s	6925	TEST SET EVALUATION
9336.0s	6926	=======================================================
9336.0s	6927	Accuracy:    55.70%
9336.0s	6928	Macro F1:    0.5376
9336.0s	6929	Weighted F1: 0.5493
9336.0s	6930	pred_count:  [360, 63, 430, 974, 591, 421, 750]
9336.0s	6931	
9336.0s	6932	Classification report:
9336.0s	6933	Angry          precision=0.4833 recall=0.3544 f1=0.4089 support=491
9336.0s	6934	Disgust        precision=0.5238 recall=0.6000 f1=0.5593 support=55
9336.0s	6935	Fear           precision=0.4186 recall=0.3409 f1=0.3758 support=528
9336.0s	6936	Happy          precision=0.7166 recall=0.7941 f1=0.7534 support=879
9336.0s	6937	Sad            precision=0.4078 recall=0.4057 f1=0.4068 support=594
9336.0s	6938	Surprise       precision=0.7126 recall=0.7212 f1=0.7168 support=416
9336.0s	6939	Neutral        precision=0.4973 recall=0.5958 f1=0.5422 support=626
9336.0s	6940	accuracy       0.5570
9336.0s	6941	macro avg      precision=0.5372 recall=0.5446 f1=0.5376 support=3589
9336.0s	6942	weighted avg   precision=0.5481 recall=0.5570 f1=0.5493 support=3589
9336.0s	6943	Confusion matrix saved: /kaggle/working/outputs/d6b_class_part_graph_motif_long120/evaluation/confusion_matrix.png
9336.0s	6944	Classification report saved: /kaggle/working/outputs/d6b_class_part_graph_motif_long120/evaluation/classification_report.txt
9336.0s	6945	Correct examples saved: /kaggle/working/outputs/d6b_class_part_graph_motif_long120/evaluation/correct_examples.png
9336.0s	6946	Wrong examples saved: /kaggle/working/outputs/d6b_class_part_graph_motif_long120/evaluation/wrong_examples.png
9336.0s	6947	Evaluation outputs: /kaggle/working/outputs/d6b_class_part_graph_motif_long120/evaluation
9336.0s	6948	
9336.0s	6949	$ /usr/bin/python3 scripts/visualize_d6.py --config configs/experiments/d6b_class_part_graph_motif_long120.yaml --environment kaggle --checkpoint /kaggle/working/outputs/d6b_class_part_graph_motif_long120/checkpoints/best.pth --graph_repo_path /kaggle/input/datasets/irthn1311/graph-repo/graph_repo --max_samples 32 --device cuda:0 --chunk_cache_size 8
9440.4s	6950	--- torch version: 2.10.0+cu128
9440.4s	6951	--- cuda available: True
9440.4s	6952	--- cuda device count: 2
9440.4s	6953	--- selected device: cuda:0
9440.4s	6954	--- gpu name: Tesla T4
9440.4s	6955	--- current cuda device: 0
9440.4s	6956	[FullGraphDataset test]
9440.4s	6957	chunk_cache_size=8
9440.4s	6958	num_chunks=8
9440.4s	6959	num_samples=3589
9440.4s	6960	[FullGraphDataset test] chunk cache enabled (max_chunks=8)
9440.4s	6961	[DataLoader split=test] batch_size=32 num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=None chunk_aware_shuffle=False
9440.4s	6962	D6 part masks: /kaggle/working/outputs/d6b_class_part_graph_motif_long120/figures/d6_part_masks
9440.4s	6963	D6 part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_long120/figures/d6_part_attention
9440.4s	6964	D6 slot summary: /kaggle/working/outputs/d6b_class_part_graph_motif_long120/figures/d6_slot_summary
9440.4s	6965	D6 class-part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_long120/figures/d6_class_part_attention
9440.4s	6966	D6 class motif maps: /kaggle/working/outputs/d6b_class_part_graph_motif_long120/figures/d6_class_motif_maps
9442.5s	6967	--- torch version: 2.10.0+cu128
9442.5s	6968	--- cuda available: True
9442.5s	6969	--- cuda device count: 2
9442.5s	6970	--- selected device: cuda:0
9442.5s	6971	--- gpu name: Tesla T4
9442.5s	6972	--- current cuda device: 0
9442.5s	6973	[FullGraphDataset test]
9442.5s	6974	chunk_cache_size=8
9442.5s	6975	num_chunks=8
9442.5s	6976	num_samples=3589
9442.5s	6977	[FullGraphDataset test] chunk cache enabled (max_chunks=8)
9442.5s	6978	[DataLoader split=test] batch_size=32 num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=None chunk_aware_shuffle=False
9442.5s	6979	D6 part masks: /kaggle/working/outputs/d6b_class_part_graph_motif_long120/figures/d6_part_masks
9442.5s	6980	D6 part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_long120/figures/d6_part_attention
9442.5s	6981	D6 slot summary: /kaggle/working/outputs/d6b_class_part_graph_motif_long120/figures/d6_slot_summary
9442.5s	6982	D6 class-part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_long120/figures/d6_class_part_attention
9442.5s	6983	D6 class motif maps: /kaggle/working/outputs/d6b_class_part_graph_motif_long120/figures/d6_class_motif_maps
9446.3s	6984	OUTPUT_DIR: /kaggle/working/outputs/d6b_class_part_graph_motif_long120
9446.3s	6985	best epoch: 119
9446.3s	6986	best val_macro_f1: 0.5213937040805061
9446.3s	6987	lr: 6.25e-05
9446.3s	6988	lr_after_scheduler: 6.25e-05
9446.3s	6989	val_diag_class_part_entropy_mean: 2.6525229437161335
9446.3s	6990	val_diag_effective_slots: 12.143227222746452
9446.3s	6991	
9446.3s	6992	TEST
9446.3s	6993	accuracy   : 0.5569796600724436
9446.3s	6994	macro_f1   : 0.5375937984680562
9446.3s	6995	weighted_f1: 0.5492840508585967
9446.3s	6996	pred_count : [360, 63, 430, 974, 591, 421, 750]
9446.3s	6997	
9446.3s	6998	 evaluation/confusion_matrix.png
9446.3s	6999	
9446.3s	7000	 figures/d6_slot_summary/slot_area.png
9446.3s	7001	
9446.3s	7002	 figures/d6_slot_summary/slot_similarity.png
9446.3s	7003	
9446.3s	7004	 figures/d6_class_part_attention/class_part_attn_grid.png
9446.3s	7005	
9446.3s	7006	 figures/d6_class_part_attention/class_part_attn_avg_by_true_class.png
9446.3s	7007	
9446.3s	7008	Class attention CSVs:
9446.3s	7009	  figures/d6_class_part_attention/class_part_attention_entropy.csv
9446.3s	7010	  figures/d6_class_part_attention/class_part_attention_similarity.csv
9446.3s	7011	  figures/d6_class_part_attention/top_slots_per_class.csv
9451.4s	7012	ZIP: /kaggle/working/outputs/d6b_class_part_graph_motif_long120.zip


d6b_class_part_graph_motif_border075.yaml
TEST SET EVALUATION
7756.9s	361	=======================================================
7756.9s	362	Accuracy:    56.06%
7756.9s	363	Macro F1:    0.5359
7756.9s	364	Weighted F1: 0.5569
7756.9s	365	pred_count:  [355, 52, 439, 962, 773, 356, 652]
7756.9s	366	
7756.9s	367	Classification report:
7756.9s	368	Angry          precision=0.4986 recall=0.3605 f1=0.4184 support=491
7756.9s	369	Disgust        precision=0.5192 recall=0.4909 f1=0.5047 support=55
7756.9s	370	Fear           precision=0.4077 recall=0.3390 f1=0.3702 support=528
7756.9s	371	Happy          precision=0.7349 recall=0.8043 f1=0.7681 support=879
7756.9s	372	Sad            precision=0.4075 recall=0.5303 f1=0.4609 support=594
7756.9s	373	Surprise       precision=0.7640 recall=0.6538 f1=0.7047 support=416
7756.9s	374	Neutral        precision=0.5138 recall=0.5351 f1=0.5243 support=626
7756.9s	375	accuracy       0.5606
7756.9s	376	macro avg      precision=0.5494 recall=0.5306 f1=0.5359 support=3589
7756.9s	377	weighted avg   precision=0.5618 recall=0.5606 f1=0.5569 support=3589
7756.9s	378	Confusion matrix saved: /kaggle/working/outputs/d6b_class_part_graph_motif_border075/evaluation/confusion_matrix.png
7756.9s	379	Classification report saved: /kaggle/working/outputs/d6b_class_part_graph_motif_border075/evaluation/classification_report.txt
7756.9s	380	Correct examples saved: /kaggle/working/outputs/d6b_class_part_graph_motif_border075/evaluation/correct_examples.png
7756.9s	381	Wrong examples saved: /kaggle/working/outputs/d6b_class_part_graph_motif_border075/evaluation/wrong_examples.png
7756.9s	382	Evaluation outputs: /kaggle/working/outputs/d6b_class_part_graph_motif_border075/evaluation
7756.9s	383	
7756.9s	384	$ /usr/bin/python3 scripts/visualize_d6.py --config configs/experiments/d6b_class_part_graph_motif_border075.yaml --environment kaggle --checkpoint /kaggle/working/outputs/d6b_class_part_graph_motif_border075/checkpoints/best.pth --graph_repo_path /kaggle/input/datasets/irthn1311/graph-repo/graph_repo --max_samples 32 --device cuda:0 --chunk_cache_size 8
7855.7s	385	--- torch version: 2.10.0+cu128
7855.7s	386	--- cuda available: True
7855.7s	387	--- cuda device count: 2
7855.7s	388	--- selected device: cuda:0
7855.7s	389	--- gpu name: Tesla T4
7855.7s	390	--- current cuda device: 0
7855.7s	391	[FullGraphDataset test]
7855.7s	392	chunk_cache_size=8
7855.7s	393	num_chunks=8
7855.7s	394	num_samples=3589
7855.7s	395	[FullGraphDataset test] chunk cache enabled (max_chunks=8)
7855.7s	396	[DataLoader split=test] batch_size=32 num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=None chunk_aware_shuffle=False
7855.7s	397	D6 part masks: /kaggle/working/outputs/d6b_class_part_graph_motif_border075/figures/d6_part_masks
7855.7s	398	D6 part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_border075/figures/d6_part_attention
7855.7s	399	D6 slot summary: /kaggle/working/outputs/d6b_class_part_graph_motif_border075/figures/d6_slot_summary
7855.7s	400	D6 class-part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_border075/figures/d6_class_part_attention
7855.7s	401	D6 class motif maps: /kaggle/working/outputs/d6b_class_part_graph_motif_border075/figures/d6_class_motif_maps
7857.4s	402	--- torch version: 2.10.0+cu128
7857.4s	403	--- cuda available: True
7857.4s	404	--- cuda device count: 2
7857.4s	405	--- selected device: cuda:0
7857.4s	406	--- gpu name: Tesla T4
7857.4s	407	--- current cuda device: 0
7857.4s	408	[FullGraphDataset test]
7857.4s	409	chunk_cache_size=8
7857.4s	410	num_chunks=8
7857.4s	411	num_samples=3589
7857.4s	412	[FullGraphDataset test] chunk cache enabled (max_chunks=8)
7857.4s	413	[DataLoader split=test] batch_size=32 num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=None chunk_aware_shuffle=False
7857.4s	414	D6 part masks: /kaggle/working/outputs/d6b_class_part_graph_motif_border075/figures/d6_part_masks
7857.4s	415	D6 part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_border075/figures/d6_part_attention
7857.4s	416	D6 slot summary: /kaggle/working/outputs/d6b_class_part_graph_motif_border075/figures/d6_slot_summary
7857.4s	417	D6 class-part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_border075/figures/d6_class_part_attention
7857.4s	418	D6 class motif maps: /kaggle/working/outputs/d6b_class_part_graph_motif_border075/figures/d6_class_motif_maps
7860.4s	419	OUTPUT_DIR: /kaggle/working/outputs/d6b_class_part_graph_motif_border075
7860.4s	420	best epoch: 93
7860.4s	421	best val_macro_f1: 0.5214340502039162
7860.4s	422	lr: 0.0005
7860.4s	423	lr_after_scheduler: 0.0005
7860.4s	424	val_diag_class_part_entropy_mean: 2.665753256958143
7860.4s	425	val_diag_effective_slots: 12.367086250170143
7860.4s	426	
7860.4s	427	TEST
7860.4s	428	accuracy   : 0.5606018389523544
7860.4s	429	macro_f1   : 0.5358819548647915
7860.4s	430	weighted_f1: 0.556948551795033
7860.4s	431	pred_count : [355, 52, 439, 962, 773, 356, 652]
7860.4s	432	
7860.4s	433	 evaluation/confusion_matrix.png
7860.4s	434	
7860.4s	435	 figures/d6_slot_summary/slot_area.png
7860.4s	436	
7860.4s	437	 figures/d6_slot_summary/slot_similarity.png
7860.4s	438	
7860.4s	439	 figures/d6_class_part_attention/class_part_attn_grid.png
7860.4s	440	
7860.4s	441	 figures/d6_class_part_attention/class_part_attn_avg_by_true_class.png
7860.4s	442	
7860.4s	443	Class attention CSVs:
7860.4s	444	  figures/d6_class_part_attention/class_part_attention_entropy.csv
7860.4s	445	  figures/d6_class_part_attention/class_part_attention_similarity.csv
7860.4s	446	  figures/d6_class_part_attention/top_slots_per_class.csv
7864.6s	447	ZIP: /kaggle/working/outputs/d6b_class_part_graph_motif_border075.zip


d6b_class_part_graph_motif_border010.yaml
8570.1s	7977	
8570.1s	7978	=======================================================
8570.1s	7979	TEST SET EVALUATION
8570.1s	7980	=======================================================
8570.1s	7981	Accuracy:    53.47%
8570.1s	7982	Macro F1:    0.4982
8570.1s	7983	Weighted F1: 0.5297
8570.1s	7984	pred_count:  [310, 80, 462, 960, 826, 419, 532]
8570.1s	7985	
8570.1s	7986	Classification report:
8570.1s	7987	Angry          precision=0.4710 recall=0.2974 f1=0.3645 support=491
8570.1s	7988	Disgust        precision=0.3375 recall=0.4909 f1=0.4000 support=55
8570.1s	7989	Fear           precision=0.3831 recall=0.3352 f1=0.3576 support=528
8570.1s	7990	Happy          precision=0.7156 recall=0.7816 f1=0.7471 support=879
8570.1s	7991	Sad            precision=0.3692 recall=0.5135 f1=0.4296 support=594
8570.1s	7992	Surprise       precision=0.6850 recall=0.6899 f1=0.6874 support=416
8570.1s	7993	Neutral        precision=0.5451 recall=0.4633 f1=0.5009 support=626
8570.1s	7994	accuracy       0.5347
8570.1s	7995	macro avg      precision=0.5009 recall=0.5102 f1=0.4982 support=3589
8570.1s	7996	weighted avg   precision=0.5368 recall=0.5347 f1=0.5297 support=3589
8570.1s	7997	Confusion matrix saved: /kaggle/working/outputs/d6b_class_part_graph_motif_border010/evaluation/confusion_matrix.png
8570.1s	7998	Classification report saved: /kaggle/working/outputs/d6b_class_part_graph_motif_border010/evaluation/classification_report.txt
8570.1s	7999	Correct examples saved: /kaggle/working/outputs/d6b_class_part_graph_motif_border010/evaluation/correct_examples.png
8570.1s	8000	Wrong examples saved: /kaggle/working/outputs/d6b_class_part_graph_motif_border010/evaluation/wrong_examples.png
8570.1s	8001	Evaluation outputs: /kaggle/working/outputs/d6b_class_part_graph_motif_border010/evaluation
8571.6s	8002	
8571.6s	8003	$ /usr/bin/python3 scripts/visualize_d6.py --config configs/experiments/d6b_class_part_graph_motif_border010.yaml --environment kaggle --checkpoint /kaggle/working/outputs/d6b_class_part_graph_motif_border010/checkpoints/best.pth --graph_repo_path /kaggle/input/datasets/irthn1311/graph-repo/graph_repo --max_samples 32 --device cuda:0 --chunk_cache_size 8
8700.8s	8004	--- torch version: 2.10.0+cu128
8700.8s	8005	--- cuda available: True
8700.8s	8006	--- cuda device count: 2
8700.8s	8007	--- selected device: cuda:0
8700.8s	8008	--- gpu name: Tesla T4
8700.8s	8009	--- current cuda device: 0
8700.8s	8010	[FullGraphDataset test]
8700.8s	8011	chunk_cache_size=8
8700.8s	8012	num_chunks=8
8700.8s	8013	num_samples=3589
8700.8s	8014	[FullGraphDataset test] chunk cache enabled (max_chunks=8)
8700.8s	8015	[DataLoader split=test] batch_size=32 num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=None chunk_aware_shuffle=False
8700.8s	8016	D6 part masks: /kaggle/working/outputs/d6b_class_part_graph_motif_border010/figures/d6_part_masks
8700.8s	8017	D6 part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_border010/figures/d6_part_attention
8700.8s	8018	D6 slot summary: /kaggle/working/outputs/d6b_class_part_graph_motif_border010/figures/d6_slot_summary
8700.8s	8019	D6 class-part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_border010/figures/d6_class_part_attention
8700.8s	8020	D6 class motif maps: /kaggle/working/outputs/d6b_class_part_graph_motif_border010/figures/d6_class_motif_maps
8702.9s	8021	--- torch version: 2.10.0+cu128
8702.9s	8022	--- cuda available: True
8702.9s	8023	--- cuda device count: 2
8702.9s	8024	--- selected device: cuda:0
8702.9s	8025	--- gpu name: Tesla T4
8702.9s	8026	--- current cuda device: 0
8702.9s	8027	[FullGraphDataset test]
8702.9s	8028	chunk_cache_size=8
8702.9s	8029	num_chunks=8
8702.9s	8030	num_samples=3589
8702.9s	8031	[FullGraphDataset test] chunk cache enabled (max_chunks=8)
8702.9s	8032	[DataLoader split=test] batch_size=32 num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=None chunk_aware_shuffle=False
8702.9s	8033	D6 part masks: /kaggle/working/outputs/d6b_class_part_graph_motif_border010/figures/d6_part_masks
8702.9s	8034	D6 part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_border010/figures/d6_part_attention
8702.9s	8035	D6 slot summary: /kaggle/working/outputs/d6b_class_part_graph_motif_border010/figures/d6_slot_summary
8702.9s	8036	D6 class-part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_border010/figures/d6_class_part_attention
8702.9s	8037	D6 class motif maps: /kaggle/working/outputs/d6b_class_part_graph_motif_border010/figures/d6_class_motif_maps
8706.8s	8038	OUTPUT_DIR: /kaggle/working/outputs/d6b_class_part_graph_motif_border010
8706.8s	8039	best epoch: 76
8706.8s	8040	best val_macro_f1: 0.5021345355577055
8706.8s	8041	lr: 0.00025
8706.8s	8042	lr_after_scheduler: 0.00025
8706.8s	8043	val_diag_class_part_entropy_mean: 2.6410724610353995
8706.8s	8044	val_diag_effective_slots: 10.956435152914672
8706.8s	8045	
8706.8s	8046	TEST
8706.8s	8047	accuracy   : 0.5346893285037615
8706.8s	8048	macro_f1   : 0.49816163387513174
8706.8s	8049	weighted_f1: 0.5297326119825204
8706.8s	8050	pred_count : [310, 80, 462, 960, 826, 419, 532]
8706.8s	8051	
8706.8s	8052	 evaluation/confusion_matrix.png
8706.8s	8053	
8706.8s	8054	 figures/d6_slot_summary/slot_area.png
8706.8s	8055	
8706.8s	8056	 figures/d6_slot_summary/slot_similarity.png
8706.8s	8057	
8706.8s	8058	 figures/d6_class_part_attention/class_part_attn_grid.png
8706.8s	8059	
8706.8s	8060	 figures/d6_class_part_attention/class_part_attn_avg_by_true_class.png
8706.8s	8061	
8706.8s	8062	Class attention CSVs:
8706.8s	8063	  figures/d6_class_part_attention/class_part_attention_entropy.csv
8706.8s	8064	  figures/d6_class_part_attention/class_part_attention_similarity.csv
8706.8s	8065	  figures/d6_class_part_attention/top_slots_per_class.csv
8712.0s	8066	ZIP: /kaggle/working/outputs/d6b_class_part_graph_motif_border010.zip




d6b_class_part_graph_motif_balance015.yaml

7851.0s	394	TEST SET EVALUATION
7851.0s	395	=======================================================
7851.0s	396	Accuracy:    53.22%
7851.0s	397	Macro F1:    0.5084
7851.0s	398	Weighted F1: 0.5328
7851.0s	399	pred_count:  [406, 78, 466, 869, 752, 406, 612]
7851.0s	400	
7851.0s	401	Classification report:
7851.0s	402	Angry          precision=0.4236 recall=0.3503 f1=0.3835 support=491
7851.0s	403	Disgust        precision=0.3846 recall=0.5455 f1=0.4511 support=55
7851.0s	404	Fear           precision=0.3948 recall=0.3485 f1=0.3702 support=528
7851.0s	405	Happy          precision=0.7514 recall=0.7429 f1=0.7471 support=879
7851.0s	406	Sad            precision=0.3524 recall=0.4461 f1=0.3938 support=594
7851.0s	407	Surprise       precision=0.7044 recall=0.6875 f1=0.6959 support=416
7851.0s	408	Neutral        precision=0.5229 recall=0.5112 f1=0.5170 support=626
7851.0s	409	accuracy       0.5322
7851.0s	410	macro avg      precision=0.5049 recall=0.5188 f1=0.5084 support=3589
7851.0s	411	weighted avg   precision=0.5372 recall=0.5322 f1=0.5328 support=3589
7851.0s	412	Confusion matrix saved: /kaggle/working/outputs/d6b_class_part_graph_motif_balance015/evaluation/confusion_matrix.png
7851.0s	413	Classification report saved: /kaggle/working/outputs/d6b_class_part_graph_motif_balance015/evaluation/classification_report.txt
7851.0s	414	Correct examples saved: /kaggle/working/outputs/d6b_class_part_graph_motif_balance015/evaluation/correct_examples.png
7851.0s	415	Wrong examples saved: /kaggle/working/outputs/d6b_class_part_graph_motif_balance015/evaluation/wrong_examples.png
7851.0s	416	Evaluation outputs: /kaggle/working/outputs/d6b_class_part_graph_motif_balance015/evaluation
7852.2s	417	
7852.2s	418	$ /usr/bin/python3 scripts/visualize_d6.py --config configs/experiments/d6b_class_part_graph_motif_balance015.yaml --environment kaggle --checkpoint /kaggle/working/outputs/d6b_class_part_graph_motif_balance015/checkpoints/best.pth --graph_repo_path /kaggle/input/datasets/irthn1311/graph-repo/graph_repo --max_samples 32 --device cuda:0 --chunk_cache_size 8
7957.8s	419	--- torch version: 2.10.0+cu128
7957.8s	420	--- cuda available: True
7957.8s	421	--- cuda device count: 2
7957.8s	422	--- selected device: cuda:0
7957.8s	423	--- gpu name: Tesla T4
7957.8s	424	--- current cuda device: 0
7957.8s	425	[FullGraphDataset test]
7957.8s	426	chunk_cache_size=8
7957.8s	427	num_chunks=8
7957.8s	428	num_samples=3589
7957.8s	429	[FullGraphDataset test] chunk cache enabled (max_chunks=8)
7957.8s	430	[DataLoader split=test] batch_size=32 num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=None chunk_aware_shuffle=False
7957.8s	431	D6 part masks: /kaggle/working/outputs/d6b_class_part_graph_motif_balance015/figures/d6_part_masks
7957.8s	432	D6 part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_balance015/figures/d6_part_attention
7957.8s	433	D6 slot summary: /kaggle/working/outputs/d6b_class_part_graph_motif_balance015/figures/d6_slot_summary
7957.8s	434	D6 class-part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_balance015/figures/d6_class_part_attention
7957.8s	435	D6 class motif maps: /kaggle/working/outputs/d6b_class_part_graph_motif_balance015/figures/d6_class_motif_maps
7959.6s	436	--- torch version: 2.10.0+cu128
7959.6s	437	--- cuda available: True
7959.6s	438	--- cuda device count: 2
7959.6s	439	--- selected device: cuda:0
7959.6s	440	--- gpu name: Tesla T4
7959.6s	441	--- current cuda device: 0
7959.6s	442	[FullGraphDataset test]
7959.6s	443	chunk_cache_size=8
7959.6s	444	num_chunks=8
7959.6s	445	num_samples=3589
7959.6s	446	[FullGraphDataset test] chunk cache enabled (max_chunks=8)
7959.6s	447	[DataLoader split=test] batch_size=32 num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=None chunk_aware_shuffle=False
7959.6s	448	D6 part masks: /kaggle/working/outputs/d6b_class_part_graph_motif_balance015/figures/d6_part_masks
7959.6s	449	D6 part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_balance015/figures/d6_part_attention
7959.6s	450	D6 slot summary: /kaggle/working/outputs/d6b_class_part_graph_motif_balance015/figures/d6_slot_summary
7959.6s	451	D6 class-part attention: /kaggle/working/outputs/d6b_class_part_graph_motif_balance015/figures/d6_class_part_attention
7959.6s	452	D6 class motif maps: /kaggle/working/outputs/d6b_class_part_graph_motif_balance015/figures/d6_class_motif_maps
7962.7s	453	OUTPUT_DIR: /kaggle/working/outputs/d6b_class_part_graph_motif_balance015
7962.7s	454	best epoch: 95
7962.7s	455	best val_macro_f1: 0.5120271511961604
7962.7s	456	lr: 0.000125
7962.7s	457	lr_after_scheduler: 0.000125
7962.7s	458	val_diag_class_part_entropy_mean: 2.6614587602362167
7962.7s	459	val_diag_effective_slots: 12.444233978744101
7962.7s	460	
7962.7s	461	TEST
7962.7s	462	accuracy   : 0.5321816662022848
7962.7s	463	macro_f1   : 0.5083678814372676
7962.7s	464	weighted_f1: 0.5328265360421731
7962.7s	465	pred_count : [406, 78, 466, 869, 752, 406, 612]
7962.7s	466	
7962.7s	467	 evaluation/confusion_matrix.png
7962.7s	468	
7962.7s	469	 figures/d6_slot_summary/slot_area.png
7962.7s	470	
7962.7s	471	 figures/d6_slot_summary/slot_similarity.png
7962.7s	472	
7962.7s	473	 figures/d6_class_part_attention/class_part_attn_grid.png
7962.7s	474	
7962.7s	475	 figures/d6_class_part_attention/class_part_attn_avg_by_true_class.png
7962.7s	476	
7962.7s	477	Class attention CSVs:
7962.7s	478	  figures/d6_class_part_attention/class_part_attention_entropy.csv
7962.7s	479	  figures/d6_class_part_attention/class_part_attention_similarity.csv
7962.7s	480	  figures/d6_class_part_attention/top_slots_per_class.csv
7967.2s	481	ZIP: /kaggle/working/outputs/d6b_class_part_graph_motif_balance015.zip