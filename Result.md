configs/experiments/d6c_class_attended_objectives_no_supcon.yaml

=======================================================
10189.6s	7423	TEST SET EVALUATION
10189.6s	7424	=======================================================
10189.6s	7425	Accuracy:    55.48%
10189.6s	7426	Macro F1:    0.5133
10189.6s	7427	Weighted F1: 0.5473
10189.6s	7428	pred_count:  [313, 85, 495, 990, 551, 381, 774]
10189.6s	7429	
10189.6s	7430	Classification report:
10189.6s	7431	Angry          precision=0.4728 recall=0.3014 f1=0.3682 support=491
10189.6s	7432	Disgust        precision=0.3294 recall=0.5091 f1=0.4000 support=55
10189.6s	7433	Fear           precision=0.4101 recall=0.3845 f1=0.3969 support=528
10189.6s	7434	Happy          precision=0.7222 recall=0.8134 f1=0.7651 support=879
10189.6s	7435	Sad            precision=0.4446 recall=0.4125 f1=0.4279 support=594
10189.6s	7436	Surprise       precision=0.7375 recall=0.6755 f1=0.7051 support=416
10189.6s	7437	Neutral        precision=0.4793 recall=0.5927 f1=0.5300 support=626
10189.6s	7438	accuracy       0.5548
10189.6s	7439	macro avg      precision=0.5137 recall=0.5270 f1=0.5133 support=3589
10189.6s	7440	weighted avg   precision=0.5496 recall=0.5548 f1=0.5473 support=3589
10189.6s	7441	Confusion matrix saved: /kaggle/working/outputs/d6c_class_attended_objectives_no_supcon/evaluation/confusion_matrix.png
10189.6s	7442	Classification report saved: /kaggle/working/outputs/d6c_class_attended_objectives_no_supcon/evaluation/classification_report.txt
10189.6s	7443	Correct examples saved: /kaggle/working/outputs/d6c_class_attended_objectives_no_supcon/evaluation/correct_examples.png
10189.6s	7444	Wrong examples saved: /kaggle/working/outputs/d6c_class_attended_objectives_no_supcon/evaluation/wrong_examples.png
10189.6s	7445	Evaluation outputs: /kaggle/working/outputs/d6c_class_attended_objectives_no_supcon/evaluation
10189.6s	7446	
10189.6s	7447	$ /usr/bin/python3 scripts/visualize_d6.py --config configs/experiments/d6c_class_attended_objectives_no_supcon.yaml --environment kaggle --checkpoint /kaggle/working/outputs/d6c_class_attended_objectives_no_supcon/checkpoints/best.pth --graph_repo_path /kaggle/input/datasets/irthn1311/graph-repo/graph_repo --max_samples 32 --device cuda:0 --chunk_cache_size 8
10309.9s	7448	--- torch version: 2.10.0+cu128
10309.9s	7449	--- cuda available: True
10309.9s	7450	--- cuda device count: 2
10309.9s	7451	--- selected device: cuda:0
10309.9s	7452	--- gpu name: Tesla T4
10309.9s	7453	--- current cuda device: 0
10309.9s	7454	[FullGraphDataset test]
10309.9s	7455	chunk_cache_size=8
10309.9s	7456	num_chunks=8
10309.9s	7457	num_samples=3589
10309.9s	7458	[FullGraphDataset test] chunk cache enabled (max_chunks=8)
10309.9s	7459	[DataLoader split=test] batch_size=32 num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=None chunk_aware_shuffle=False
10309.9s	7460	D6 part masks: /kaggle/working/outputs/d6c_class_attended_objectives_no_supcon/figures/d6_part_masks
10309.9s	7461	D6 part attention: /kaggle/working/outputs/d6c_class_attended_objectives_no_supcon/figures/d6_part_attention
10309.9s	7462	D6 slot summary: /kaggle/working/outputs/d6c_class_attended_objectives_no_supcon/figures/d6_slot_summary
10309.9s	7463	D6 class-part attention: /kaggle/working/outputs/d6c_class_attended_objectives_no_supcon/figures/d6_class_part_attention
10309.9s	7464	D6 class motif maps: /kaggle/working/outputs/d6c_class_attended_objectives_no_supcon/figures/d6_class_motif_maps
10312.3s	7465	--- torch version: 2.10.0+cu128
10312.3s	7466	--- cuda available: True
10312.3s	7467	--- cuda device count: 2
10312.3s	7468	--- selected device: cuda:0
10312.3s	7469	--- gpu name: Tesla T4
10312.3s	7470	--- current cuda device: 0
10312.3s	7471	[FullGraphDataset test]
10312.3s	7472	chunk_cache_size=8
10312.3s	7473	num_chunks=8
10312.3s	7474	num_samples=3589
10312.3s	7475	[FullGraphDataset test] chunk cache enabled (max_chunks=8)
10312.3s	7476	[DataLoader split=test] batch_size=32 num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=None chunk_aware_shuffle=False
10312.3s	7477	D6 part masks: /kaggle/working/outputs/d6c_class_attended_objectives_no_supcon/figures/d6_part_masks
10312.3s	7478	D6 part attention: /kaggle/working/outputs/d6c_class_attended_objectives_no_supcon/figures/d6_part_attention
10312.3s	7479	D6 slot summary: /kaggle/working/outputs/d6c_class_attended_objectives_no_supcon/figures/d6_slot_summary
10312.3s	7480	D6 class-part attention: /kaggle/working/outputs/d6c_class_attended_objectives_no_supcon/figures/d6_class_part_attention
10312.3s	7481	D6 class motif maps: /kaggle/working/outputs/d6c_class_attended_objectives_no_supcon/figures/d6_class_motif_maps
10316.6s	7482	OUTPUT_DIR: /kaggle/working/outputs/d6c_class_attended_objectives_no_supcon
10316.6s	7483	best epoch: 110
10316.6s	7484	best val_macro_f1: 0.5159546636646634
10316.6s	7485	lr: 6.25e-05
10316.6s	7486	lr_group_0: 6.25e-05
10316.6s	7487	lr_after_scheduler: 6.25e-05
10316.6s	7488	val_diag_class_part_entropy_mean: 5.234923341632944
10316.6s	7489	val_diag_effective_slots: 10.085382832890064
10316.6s	7490	train_loss_class_border: 0.24365752768399795
10316.6s	7491	train_loss_class_attn_sep: 0.05018743571052095
10316.6s	7492	train_loss_supcon: 0.0
10316.6s	7493	val_loss_class_border: 0.24521723020393238
10316.6s	7494	val_loss_class_attn_sep: 0.043287024033808075
10316.6s	7495	val_loss_supcon: 0.0
10316.6s	7496	val_diag_true_class_border_mass_mean: 0.24521723020393238
10316.6s	7497	val_diag_true_class_border_mass_max: 0.2726297518320843
10316.6s	7498	val_diag_class_attn_sim_fear_sad: 0.9836572946700375
10316.6s	7499	val_diag_class_attn_sim_fear_neutral: 0.9448475853531761
10316.6s	7500	val_diag_class_attn_sim_fear_surprise: 0.9217030812153774
10316.6s	7501	val_diag_class_attn_sim_sad_neutral: 0.9655411776188201
10316.6s	7502	val_diag_class_attn_sim_angry_disgust: 0.8085158789052372
10316.6s	7503	val_diag_supcon_valid_anchors: 0.0
10316.6s	7504	val_diag_supcon_positive_pairs: 0.0
10316.6s	7505	
10316.6s	7506	TEST
10316.6s	7507	accuracy   : 0.5547506269155754
10316.6s	7508	macro_f1   : 0.5133197247659952
10316.6s	7509	weighted_f1: 0.5472754444907704
10316.6s	7510	pred_count : [313, 85, 495, 990, 551, 381, 774]
10316.6s	7511	
10316.6s	7512	 evaluation/confusion_matrix.png
10316.6s	7513	
10316.6s	7514	 figures/d6_slot_summary/slot_area.png
10316.6s	7515	
10316.6s	7516	 figures/d6_slot_summary/slot_similarity.png
10316.6s	7517	

configs/experiments/d6c_class_attended_objectives_no_sep.yaml
TEST SET EVALUATION
9641.1s	7233	=======================================================
9641.1s	7234	Accuracy:    55.64%
9641.1s	7235	Macro F1:    0.5212
9641.1s	7236	Weighted F1: 0.5490
9641.1s	7237	pred_count:  [405, 65, 434, 941, 544, 488, 712]
9641.1s	7238	
9641.1s	7239	Classification report:
9641.1s	7240	Angry          precision=0.5062 recall=0.4175 f1=0.4576 support=491
9641.1s	7241	Disgust        precision=0.4000 recall=0.4727 f1=0.4333 support=55
9641.1s	7242	Fear           precision=0.4147 recall=0.3409 f1=0.3742 support=528
9641.1s	7243	Happy          precision=0.7343 recall=0.7861 f1=0.7593 support=879
9641.1s	7244	Sad            precision=0.4136 recall=0.3788 f1=0.3954 support=594
9641.1s	7245	Surprise       precision=0.6475 recall=0.7596 f1=0.6991 support=416
9641.1s	7246	Neutral        precision=0.4972 recall=0.5655 f1=0.5291 support=626
9641.1s	7247	accuracy       0.5564
9641.1s	7248	macro avg      precision=0.5162 recall=0.5316 f1=0.5212 support=3589
9641.1s	7249	weighted avg   precision=0.5465 recall=0.5564 f1=0.5490 support=3589
9641.1s	7250	Confusion matrix saved: /kaggle/working/outputs/d6c_class_attended_objectives_no_sep/evaluation/confusion_matrix.png
9641.1s	7251	Classification report saved: /kaggle/working/outputs/d6c_class_attended_objectives_no_sep/evaluation/classification_report.txt
9641.1s	7252	Correct examples saved: /kaggle/working/outputs/d6c_class_attended_objectives_no_sep/evaluation/correct_examples.png
9641.1s	7253	Wrong examples saved: /kaggle/working/outputs/d6c_class_attended_objectives_no_sep/evaluation/wrong_examples.png
9641.1s	7254	Evaluation outputs: /kaggle/working/outputs/d6c_class_attended_objectives_no_sep/evaluation
9641.1s	7255	
9641.1s	7256	$ /usr/bin/python3 scripts/visualize_d6.py --config configs/experiments/d6c_class_attended_objectives_no_sep.yaml --environment kaggle --checkpoint /kaggle/working/outputs/d6c_class_attended_objectives_no_sep/checkpoints/best.pth --graph_repo_path /kaggle/input/datasets/irthn1311/graph-repo/graph_repo --max_samples 32 --device cuda:0 --chunk_cache_size 8
9735.6s	7257	--- torch version: 2.10.0+cu128
9735.6s	7258	--- cuda available: True
9735.6s	7259	--- cuda device count: 2
9735.6s	7260	--- selected device: cuda:0
9735.6s	7261	--- gpu name: Tesla T4
9735.6s	7262	--- current cuda device: 0
9735.6s	7263	[FullGraphDataset test]
9735.6s	7264	chunk_cache_size=8
9735.6s	7265	num_chunks=8
9735.6s	7266	num_samples=3589
9735.6s	7267	[FullGraphDataset test] chunk cache enabled (max_chunks=8)
9735.6s	7268	[DataLoader split=test] batch_size=32 num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=None chunk_aware_shuffle=False
9735.6s	7269	D6 part masks: /kaggle/working/outputs/d6c_class_attended_objectives_no_sep/figures/d6_part_masks
9735.6s	7270	D6 part attention: /kaggle/working/outputs/d6c_class_attended_objectives_no_sep/figures/d6_part_attention
9735.6s	7271	D6 slot summary: /kaggle/working/outputs/d6c_class_attended_objectives_no_sep/figures/d6_slot_summary
9735.6s	7272	D6 class-part attention: /kaggle/working/outputs/d6c_class_attended_objectives_no_sep/figures/d6_class_part_attention
9735.6s	7273	D6 class motif maps: /kaggle/working/outputs/d6c_class_attended_objectives_no_sep/figures/d6_class_motif_maps
9737.7s	7274	--- torch version: 2.10.0+cu128
9737.7s	7275	--- cuda available: True
9737.7s	7276	--- cuda device count: 2
9737.7s	7277	--- selected device: cuda:0
9737.7s	7278	--- gpu name: Tesla T4
9737.7s	7279	--- current cuda device: 0
9737.7s	7280	[FullGraphDataset test]
9737.7s	7281	chunk_cache_size=8
9737.7s	7282	num_chunks=8
9737.7s	7283	num_samples=3589
9737.7s	7284	[FullGraphDataset test] chunk cache enabled (max_chunks=8)
9737.7s	7285	[DataLoader split=test] batch_size=32 num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=None chunk_aware_shuffle=False
9737.7s	7286	D6 part masks: /kaggle/working/outputs/d6c_class_attended_objectives_no_sep/figures/d6_part_masks
9737.7s	7287	D6 part attention: /kaggle/working/outputs/d6c_class_attended_objectives_no_sep/figures/d6_part_attention
9737.7s	7288	D6 slot summary: /kaggle/working/outputs/d6c_class_attended_objectives_no_sep/figures/d6_slot_summary
9737.7s	7289	D6 class-part attention: /kaggle/working/outputs/d6c_class_attended_objectives_no_sep/figures/d6_class_part_attention
9737.7s	7290	D6 class motif maps: /kaggle/working/outputs/d6c_class_attended_objectives_no_sep/figures/d6_class_motif_maps
9741.5s	7291	OUTPUT_DIR: /kaggle/working/outputs/d6c_class_attended_objectives_no_sep
9741.5s	7292	best epoch: 109
9741.5s	7293	best val_macro_f1: 0.523310236573126
9741.5s	7294	lr: 0.000125
9741.5s	7295	lr_group_0: 0.000125
9741.5s	7296	lr_after_scheduler: 0.000125
9741.5s	7297	val_diag_class_part_entropy_mean: 4.756486555116367
9741.5s	7298	val_diag_effective_slots: 11.360884084110767
9741.5s	7299	train_loss_class_border: 0.20043346715505286
9741.5s	7300	train_loss_class_attn_sep: 0.0
9741.5s	7301	train_loss_supcon: 1.774403436139826
9741.5s	7302	val_loss_class_border: 0.19265053757524067
9741.5s	7303	val_loss_class_attn_sep: 0.0
9741.5s	7304	val_loss_supcon: 1.8711023921460177
9741.5s	7305	val_diag_true_class_border_mass_mean: 0.19265053757524067
9741.5s	7306	val_diag_true_class_border_mass_max: 0.27658189437558167
9741.5s	7307	val_diag_class_attn_sim_fear_sad: 0.8154317130029729
9741.5s	7308	val_diag_class_attn_sim_fear_neutral: 0.6912845626341558
9741.5s	7309	val_diag_class_attn_sim_fear_surprise: 0.6466473298790181
9741.5s	7310	val_diag_class_attn_sim_sad_neutral: 0.7867774710191034
9741.5s	7311	val_diag_class_attn_sim_angry_disgust: 0.8671060389122077
9741.5s	7312	val_diag_supcon_valid_anchors: 31.23008849557522
9741.5s	7313	val_diag_supcon_positive_pairs: 172.28318584070797
9741.5s	7314	
9741.5s	7315	TEST
9741.5s	7316	accuracy   : 0.5564224017832266
9741.5s	7317	macro_f1   : 0.5211681798405816
9741.5s	7318	weighted_f1: 0.5490450625370148
9741.5s	7319	pred_count : [405, 65, 434, 941, 544, 488, 712]
9741.5s	7320	
9741.5s	7321	 evaluation/confusion_matrix.png
9741.5s	7322	
9741.5s	7323	 figures/d6_slot_summary/slot_area.png
9741.5s	7324	
9741.5s	7325	 figures/d6_slot_summary/slot_similarity.png
9741.5s	7326	
9741.5s	7327	 figures/d6_class_part_attention/class_part_attn_grid.png
9741.5s	7328	
9741.5s	7329	 figures/d6_class_part_attention/class_part_attn_avg_by_true_class.png
9741.5s	7330	
9741.5s	7331	 figures/d6_class_motif_maps/class_pixel_motif_trueclass_avg.png
9741.5s	7332	
9741.5s	7333	 figures/d6_class_motif_maps/class_pixel_motif_predclass_avg.png
9741.5s	7334	
9741.5s	7335	Class attention CSVs:
9741.5s	7336	  figures/d6_class_part_attention/class_part_attention_entropy.csv
9741.5s	7337	  figures/d6_class_part_attention/class_part_attention_similarity.csv
9741.5s	7338	  figures/d6_class_part_attention/confusion_pair_attention_similarity.csv
9741.5s	7339	  figures/d6_class_part_attention/top_slots_per_class.csv
9741.5s	7340	
9741.5s	7341	Class motif CSVs:
9741.5s	7342	  figures/d6_class_motif_maps/class_motif_border_mass_by_class.csv
9741.5s	7343	  figures/d6_class_motif_maps/true_class_border_mass_by_class.csv
9746.5s	7344	ZIP: /kaggle/working/outputs/d6c_class_attended_objectives_no_sep.zip


configs/experiments/d6c_class_attended_objectives_border_only.yaml