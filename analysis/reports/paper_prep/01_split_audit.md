# split-audit (paper-prep workflow wf_f3c16087-17b, 2026-11-04)

## Özet

SPLIT-PROTOCOL AUDIT. Scope: 26 radar/mmWave papers (7 old-manuscript citations, 19 others analysed), plus 1 benchmark paper used only as corroboration (UniMM-HAR/DAP) and 2 non-radar methodology papers (Dehghani 2019, Tello 2023). Surveys and non-radar items in cas-refs.bib (Kong2024, Sesyuk2024, Selvaraj2023, Khatun2023, Chung2023, Setiawan2020) are excluded. Evidence level per item: FT = full text read; ABS = abstract only; NA = paywalled, protocol unknown.

ACCESS: 16 radar papers FT (MiliPoint, MM-Fi, mRI, mmBody, HuPR, Pantomime, Tesla-Rapture, OG-PCL, Zakka-2025, mmFall, SUPER, RadMamba, milliFlow, Alhazmi; RadHAR via the authors' README plus two papers that restate its split, not the RadHAR paper itself); 4 ABS (Xia2022, Hu2025, ElHail2025, Yan2023); 7 NA (Zhang2019, Jung2021, Yu2022, Wang2023, Jin2024, Nalci2022, Liu2024/PmTrack). MDPI, IEEE and ACM pages returned 403 or paywalls; Alhazmi was read through the PubMed Central copy.

TALLY over the 19 papers with usable text (a paper counts once under its HEADLINE protocol; multi-protocol papers are noted):
(1) Random sample/frame split stated explicitly as a reported protocol: 2 (mRI S1 'random split of all data'; MM-Fi S1 'random split of all video samples'). Both ALSO report a cross-subject protocol, and MM-Fi also cross-environment.
(2) Ratio or k-fold stated, split criterion (random vs subject vs session) NOT stated: 6 (MiliPoint 80/10/10; RadHAR fixed train/test folders; OG-PCL 5-fold CV; Zakka-2025 inherits RadHAR/MiliPoint splits; Alhazmi 80/20 train/validation; Yan2023 source/target domain setup). RadMamba is partly here (DIAT, CI4R) and partly cross-subject (UoG2020).
(3) Cross-subject / unseen-user / leave-one-subject-out as headline: 5 (Pantomime, SUPER, milliFlow, Xia2022 [ABS], Hu2025 [ABS]); plus as a secondary protocol: mRI S2, MM-Fi S2, RadMamba-UoG2020, mmBody.
(4) Cross-environment/scene as headline or co-protocol: ElHail2025 [ABS], Tesla-Rapture (via Pantomime), Pantomime, MM-Fi S3, mmBody.
(5) Whole-sequence/video-level hold-out: HuPR (193/21/21 one-minute videos, 5 of 6 subjects in both train and test), mmBody.
(6) Separate-recording (cross-session-like) hold-out: 1 (mmFall, anomaly-detection setting, only two subjects, both authors).
(7) Not accessible: 7. Unclear: all of group (2).

OVERLAPPING WINDOWS: Overlap by construction is confirmed in MiliPoint (stacks 'contiguous frames before and after the current frame'), mRI (3-frame fusion), MM-Fi (0.5 s adjacent-frame aggregation), RadHAR/Zakka-2025 (60-frame windows, 10-frame stride), SUPER (20-frame sliding window at inference) and HuPR (sliding window over frames). Whether overlapping windows were RANDOMLY split across train and test is NOT stated verbatim by any paper. Strongest candidates: mRI S1 (random split of all data, 3-frame-fused frames; unit presumably frames but not stated) and MiliPoint (ratio-only 80/10/10 over stacked data points; likely random but unstated). MM-Fi S1 splits whole video samples, so cross-set leakage is unlikely there. RadHAR's train/test folders hold separate recordings, so the 10-frame-stride overlap is within each partition, but whether subjects or sessions are shared across folders is unstated (README gives no subject count; Tesla-Rapture's Table 1 lists 2 participants, 1 environment). SUPER (LOSO) and Pantomime (unseen users) avoid cross-set leakage.

WHAT WE CAN HONESTLY WRITE: (a) 'Random sample-level splits are a commonly reported protocol in public mmWave point-cloud HAR/pose benchmarks' is supported (mRI, MM-Fi, and the UniMM-HAR statement that early point-cloud HAR datasets 'relied on random splits'). (b) 'Many works do not state whether overlapping windows are separated by recording, session or subject before splitting' is supported (6-7 of 19). (c) 'Random frame/window splits are the standard' is NOT supported: roughly 10 of 19 papers use or also report cross-subject, cross-environment or sequence-level protocols, and MM-Fi reports only about 3 mm PA-MPJPE change from random to cross-subject for pose. (d) The mechanism and magnitude of leakage come from non-radar HAR (Dehghani 2019: +10% generally, +16% with overlapping windows; Tello 2023: sliding windows plus random k-fold common and biased); cite them as such, not as radar evidence. (e) No paper in this set re-evaluates a radar result under a leakage-free protocol, which supports the novelty of our cycle-grouped re-evaluation but is an absence-of-evidence statement for papers not read (the 7 NA papers).

CAUTIONS: Verbatim quotes are from extracted PDF text; wrapped-line hyphenation was repaired. Years follow the document's own arXiv/venue line. RadHAR-paper's own text was not read. Zhang2019 etc. are marked NA rather than inferred from abstracts.

## Kalemler

### 1. Cui H., Zhong S., Wu J., Shen Z., Dahnoun N., Zhao Y. (2023). MiliPoint: A Point Cloud Dataset for mmWave Radar. arXiv:2309.13425 (v2, Nov 2023).
- **DOI/URL:** https://arxiv.org/abs/2309.13425
- **Bulgu:** PROTOCOL: ratio-only 80/10/10 train/val/test split; split criterion (random vs subject vs session) NOT stated; no cross-subject protocol described although 11 participants exist. WINDOWS OVERLAP: yes (each data point stacks contiguous frames, s=5 for identification, s=50 for action). Likely a random split of overlapping stacked samples but this is not stated, so strict category = unclear (ratio-only). FT.
- **Alıntı:** "with a split of 80%, 10% and 10% for training, validation, and testing partitions, respectively ... the stack will pile up contiguous frames both before and after the current frame"
- **Kaynak:** https://arxiv.org/pdf/2309.13425
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 2. Yang J., Huang H., Zhou Y., Chen X., Xu Y., Yuan S., Zou H., Lu C. X., Xie L. (2023). MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing. arXiv:2305.10345 (v2, Sep 2023).
- **DOI/URL:** https://arxiv.org/abs/2305.10345
- **Bulgu:** PROTOCOL: three provided splits. S1 = random split of all video samples (sequence/sample-level, seen subjects); S2 = cross-subject (32 train / 8 test); S3 = cross-environment (3 train / 1 test). WINDOWS OVERLAP: mmWave frames aggregate adjacent frames within 0.5 s (within-sequence overlap), but S1 splits whole video samples so cross-set leakage is unlikely. Paper notes PA-MPJPE changes by only about 3 mm between S1 and S2 for LiDAR/mmWave. FT.
- **Alıntı:** "Setting 1 (S1 Random Split) involves a random split of all video samples into training and testing sets, with a split ratio of 3:1"
- **Kaynak:** https://arxiv.org/pdf/2305.10345
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 3. An S., Li Y., Ogras U. (2022). mRI: Multi-modal 3D Human Pose Estimation Dataset using mmWave, RGB-D, and Inertial Sensors. arXiv:2210.08394.
- **DOI/URL:** https://arxiv.org/abs/2210.08394
- **Bulgu:** PROTOCOL: two settings. S1 = random 80/20 split of all data (frame-level, seen subjects, personalised); S2 = split by subject (16 train / 4 test subjects). Action detection reported under S2 only. WINDOWS OVERLAP: yes, three consecutive frames are fused into each sample, so with a frame-level S1 neighbouring overlapping samples can fall on both sides (the unit of S1 is not stated explicitly, so leakage is probable, not confirmed). FT.
- **Alıntı:** "Setting 1 (S1 Random Split): A random split of 80% and 20% of all data is used as the training and testing set"
- **Kaynak:** https://arxiv.org/pdf/2210.08394
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 4. Chen A., Wang X., Zhu S., Li Y., Chen J., Ye Q. (2022). mmBody Benchmark: 3D Body Reconstruction Dataset and Analysis for Millimeter Wave Radar. Proc. ACM Multimedia (MM '22), Lisboa; arXiv:2209.05070.
- **DOI/URL:** https://arxiv.org/abs/2209.05070
- **Bulgu:** PROTOCOL: sequence-level hold-out. Train = 20 lab sequences from 10 subjects; test = held-out sequences, mostly from different subjects and different scenes (rain, smoke, poor lighting, occlusion), with one test scene (Lab2) containing seen subjects. Category = cross-subject/cross-scene (mostly). No frame-level random split. FT.
- **Alıntı:** "except Lab2 containing seen subjects in the training, 2 sequences from different subjects for each scene are held out"
- **Kaynak:** https://arxiv.org/pdf/2209.05070
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 5. Lee S.-P., Kini N. P., Peng W.-H., Ma C.-W., Hwang J.-N. (2022). HuPR: A Benchmark for Human Pose Estimation Using Millimeter Wave Radar. arXiv:2210.12564.
- **DOI/URL:** https://arxiv.org/abs/2210.12564
- **Bulgu:** PROTOCOL: whole-video-sequence split (193 train / 21 val / 21 test one-minute sequences); selection criterion for the sequence split not stated; 6 subjects of which only one appears solely in the test set, so 5 subjects are seen in training. Category = sequence-level hold-out, mostly seen subjects (not frame-level random). WINDOWS OVERLAP: a sliding window of radar frames is used as input, but windows from one video do not cross partitions. FT.
- **Alıntı:** "We take 193 video sequences for training, 21 for validation, and 21 for test"
- **Kaynak:** https://arxiv.org/pdf/2210.12564
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 6. Palipana S., Salami D., Leiva L. A., Sigg S. (2021). Pantomime: Mid-Air Gesture Recognition with Sparse Millimeter-Wave Radar Point Clouds. Proc. ACM IMWUT 5(1), Article 27 (March 2021).
- **DOI/URL:** https://doi.org/10.1145/3448110
- **Bulgu:** PROTOCOL: cross-user (and cross-environment). Although called a 'user-dependent 60:20:20 ratio', the test partition holds 8 users unseen in training/validation; further experiments train on Open+Office and test on other environments, distances, angles, speeds. Category = cross-user + cross-environment. Whole gesture instances are the samples (no sliding-window split). FT (author postprint).
- **Alıntı:** "the remaining 20% of the data (data from 8 users unseen by the training and validation sets) is used for testing"
- **Kaynak:** https://orbilu.uni.lu/bitstream/10993/46834/1/pantomime.pdf
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 7. Singh A. D., Sandha S. S., Garcia L., Srivastava M. (2019). RadHAR: Human Activity Recognition from Point Clouds Generated through a Millimeter-wave Radar. Proc. 3rd ACM Workshop on Millimeter-wave Networks and Sensing Systems (mmNets), pp. 51-56.
- **DOI/URL:** https://github.com/nesl/RadHAR
- **Bulgu:** PROTOCOL: fixed train/test folders of separate recordings (71.6 min train / 21.4 min test; 12,097 train / 3,538 test windows). The criterion that separates train and test files (subject, session, time) is NOT stated in the README; Tesla-Rapture's table lists 2 participants, 1 environment. Category = unclear (fixed file-level split). WINDOWS OVERLAP: yes, 60-frame windows with a 10-frame stride, overlapping within each partition. The RadHAR paper itself was not read (only the authors' README and papers restating the split). FT-partial.
- **Alıntı:** "Windows of 2 seconds (60 frames) having a sliding factor of 0.33 seconds (10 frames). ... The data is partitioned into the folders train and test."
- **Kaynak:** https://github.com/nesl/RadHAR
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 8. Salami D., Hasibi R., Palipana S., Popovski P., Michoel T., Sigg S. (2021). Tesla-Rapture: A Lightweight Gesture Recognition System from mmWave Radar Point Clouds. arXiv:2109.06448 (published later in IEEE Trans. Mobile Comput., DOI 10.1109/TMC.2022.3153717 as shown in the PDF link annotation).
- **DOI/URL:** https://arxiv.org/abs/2109.06448
- **Bulgu:** PROTOCOL: reuses the dataset authors' own splits for Pantomime (cross-user) and RadHAR (fixed folders, 2 participants, 1 environment); cross-environment evaluation on Pantomime (train Open+Office, test five other environments). Category = inherited (cross-user/cross-environment for Pantomime; unclear fixed split for RadHAR). Also useful as third-party description of RadHAR (Table 1: 2 participants, 1 environment, 12,097 train samples). FT.
- **Alıntı:** "we employ the same train, validation, test splits provided by Pantomime and RadHAR authors, respectively"
- **Kaynak:** https://arxiv.org/pdf/2109.06448
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 9. Yan J., Wang Y., Xu C., Liu D. (2025). OG-PCL: Efficient Sparse Point Cloud Processing for Human Activity Recognition. arXiv:2511.08910.
- **DOI/URL:** https://arxiv.org/abs/2511.08910
- **Bulgu:** PROTOCOL: 5-fold cross-validation on RadHAR (200 sequences of T=60 frames, 5 classes); whether folds are grouped by recording/subject/session, and whether the sequences come from overlapping windows, is NOT stated. Category = unclear (k-fold, grouping unstated); overlap unclear. FT.
- **Alıntı:** "The model was trained using 5-fold cross-validation"
- **Kaynak:** https://arxiv.org/pdf/2511.08910
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 10. Zakka V. G., Manso L. J., Dai Z. (2025). Multi-Head Adaptive Graph Convolution Network for Sparse Point Cloud-Based Human Activity Recognition. arXiv:2504.02778.
- **DOI/URL:** https://arxiv.org/abs/2504.02778
- **Bulgu:** PROTOCOL: inherits the RadHAR ('MMActivity') fixed train/test split plus 20% of the training data held out for validation, and an 80/10/10 split of MiliPoint 'following the approach in' its authors; criterion unstated. Category = unclear (inherited ratio/file split). WINDOWS OVERLAP: yes (60-frame windows, 10-frame stride); the 20% validation hold-out is drawn from these overlapping training windows (unit not stated). FT.
- **Alıntı:** "2-second windows (60 frames) were created with a 10-frame sliding window, yielding 12,097 training samples and 3,538 test samples"
- **Kaynak:** https://arxiv.org/pdf/2504.02778
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 11. Jin F., Sengupta A., Cao S. (2020). mmFall: Fall Detection using 4D MmWave Radar and a Hybrid Variational RNN AutoEncoder. arXiv:2003.02386 (preprint v4, Jul 2020).
- **DOI/URL:** https://arxiv.org/abs/2003.02386
- **Bulgu:** PROTOCOL: anomaly-detection setting closest to our own. Train on normal activity only (DS0, about 2 h of normal ADL), test on separately recorded datasets (DS1, DS2) containing unseen falls/jumps; 10-frame (1 s at 10 fps) windows. Category = separate-recording hold-out (cross-session-like). Only two subjects, both the first two authors, so no cross-subject evidence. FT.
- **Alıntı:** "we trained these three models on the normal dataset DS0, and then tested on dataset DS1 in which there are some normal motions"
- **Kaynak:** https://arxiv.org/pdf/2003.02386
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 12. Zhang B., Zhou Z., Jiang B., Zheng R. (2024). SUPER: Seated Upper Body Pose Estimation using mmWave Radars. arXiv:2407.02455.
- **DOI/URL:** https://arxiv.org/abs/2407.02455
- **Bulgu:** PROTOCOL: leave-one-subject-out (10 participants, 30 trials). Category = cross-subject (LOSO). WINDOWS OVERLAP: input processed with a 20-frame sliding window, but LOSO keeps all windows of a subject on one side, so no cross-set leakage. FT.
- **Alıntı:** "PCK@15mm of the 14 upper body joints in leave-one-subject-out experiments"
- **Kaynak:** https://arxiv.org/pdf/2407.02455
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 13. Wu Y., Fioranelli F., Gao C. (2026). RadMamba: Efficient Human Activity Recognition through Radar-based Micro-Doppler-Oriented Mamba State-Space Model. IEEE Trans. Radar Systems 4:261-272 (cited as Wu2025 in cas-refs.bib; arXiv:2504.12039). Micro-Doppler, not point cloud.
- **DOI/URL:** https://doi.org/10.1109/TRS.2025.3648848
- **Bulgu:** PROTOCOL: three datasets, three protocols. DIAT: 70/10/20 per-class split of 3,780 discrete 3-s spectrogram samples (criterion beyond 'within each class' unstated; 30 subjects not stated to be separated). CI4R (IWR1443, 6 participants, 10 repetitions per activity): 8:2 train-to-test, criterion unstated. UoG2020: 13 participants train, 2 held out (cross-subject). Category = mixed: unclear-ratio (2 datasets) + cross-subject (1). Samples are discrete, not overlapping sliding windows. FT.
- **Alıntı:** "data from two participants designated as the test set and data from the remaining 13 participants used for training"
- **Kaynak:** https://arxiv.org/pdf/2504.12039
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 14. Ding F., Luo Z., Zhao P., Lu C. X. (2025). milliFlow: Scene Flow Estimation on mmWave Radar Point Cloud for Human Motion Sensing. ECCV 2024, LNCS 15082, pp. 202-221 (cited as Ding2025; arXiv:2306.17010).
- **DOI/URL:** https://doi.org/10.1007/978-3-031-72691-0_12
- **Bulgu:** PROTOCOL: split by subject, 12 participants, train:val:test = 3:1:2, plus an out-of-set test with unseen activities; HAR evaluated with 20-frame input sequences. Category = cross-subject. Sequences cropped to 200 frames per activity/scene; overlap of the 20-frame inputs not stated but all stay within a subject's partition. FT.
- **Alıntı:** "we divide the dataset into three parts by subject following the ratio of train:val:test = 3:1:2"
- **Kaynak:** https://arxiv.org/pdf/2306.17010
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 15. Alhazmi A. K. et al. (2024). Intelligent Millimeter-Wave System for Human Activity Monitoring for Telemedicine. Sensors 24(1):268.
- **DOI/URL:** https://doi.org/10.3390/s24010268
- **Bulgu:** PROTOCOL: 89 participants x 5 activities (five 30-s samples each, many frames); data split 80% training / 20% validation, with NO separate test set; split unit (frame, window or 30-s sample) and any subject separation are NOT stated. The reported 99.5% accuracy comes from this protocol. Category = unclear (ratio-only), subject independence not claimed; overlap unclear. FT (PubMed Central copy).
- **Alıntı:** "The data were split into training (80%) and validation (20%) sets to adjust hyperparameters, prevent overfitting"
- **Kaynak:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10781319/
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 16. Lin J., Wu S., Liu J., Wang C., Liu M. (2026). DAP: Doppler-aware Point Network for Heterogeneous mmWave Action Recognition (introduces the UniMM-HAR benchmark over RadHAR, mRI, MM-Fi). arXiv:2605.09604 (v2, Jul 2026).
- **DOI/URL:** https://arxiv.org/abs/2605.09604
- **Bulgu:** SECONDARY (third-party) evidence for our claim: authors state early mmWave point-cloud HAR datasets relied on random splits, and define Random (60:40 per source), Cross-Subject (5:5) and Cross-Set (4:2 by scene) protocols, i.e. a 2026 benchmark treats random splits as the conventional baseline. Not a primary protocol audit of earlier papers (it cites them only by reference number). FT.
- **Alıntı:** "Early mmWave point cloud HAR datasets were limited in scale and relied on random splits."
- **Kaynak:** https://arxiv.org/pdf/2605.09604
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 17. Xia Z. et al. (2022). Time-Space Dimension Reduction of Millimeter-Wave Radar Point-Clouds for Smart-Home Hand-Gesture Recognition. IEEE Sensors Journal (cited as Xia2022).
- **DOI/URL:** https://doi.org/10.1109/JSEN.2022.3145844
- **Bulgu:** PROTOCOL (abstract only, full text not accessible): reports separate accuracies for the trained user and two non-trained users, i.e. a cross-user evaluation alongside multi-position data. Category = cross-user (ABS). Window overlap unknown.
- **Alıntı:** "for trained user A, non-trained user B, and non-trained user C can reach 99.7%, 91.1%, and 87.1%, respectively"
- **Kaynak:** https://doi.org/10.1109/JSEN.2022.3145844
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 18. Hu et al. (2025). Human Activity Recognition Trained on Simulated Millimeter-Wave Radar Data with Domain Adaptation. IEEE Trans. Instrum. Meas. (cited as Hu2025).
- **DOI/URL:** https://doi.org/10.1109/TIM.2025.3558216
- **Bulgu:** PROTOCOL (abstract only, full text not accessible): evaluated on entirely new subjects, i.e. cross-subject. Category = cross-subject (ABS). Window overlap unknown. Micro-Doppler/FMCW simulation, not on-chip point cloud.
- **Alıntı:** "achieves up to a 99.0% recognition accuracy rate across various positional scenarios for entirely new subjects"
- **Kaynak:** https://doi.org/10.1109/TIM.2025.3558216
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 19. El Hail R., Mehrjouseresht P., Schreurs D. M. M.-P., Karsmakers P. (2025). Radar-Based Human Activity Recognition: A Study on Cross-Environment Robustness. Electronics 14(5):875.
- **DOI/URL:** https://doi.org/10.3390/electronics14050875
- **Bulgu:** PROTOCOL (abstract only, MDPI full text returned 403): cross-environment by design (same-environment vs different-environment validation sets, 29.6% F1 drop). Category = cross-environment (ABS). How the same-environment validation set is split from the training data is unknown. Notable as a paper that quantifies the generalisation gap that in-distribution splits hide.
- **Alıntı:** "a baseline HAR model was evaluated using a validation set (a) from the same environment as the training data and (b) from a different environment"
- **Kaynak:** https://api.crossref.org/works/10.3390/electronics14050875
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 20. Yan et al. (2023). mmGesture: Semi-supervised gesture recognition system using mmWave radar. Expert Systems with Applications (cited as Yan2023).
- **DOI/URL:** https://doi.org/10.1016/j.eswa.2022.119042
- **Bulgu:** PROTOCOL (abstract only, full text not accessible): domain-shift setting with few labelled source-domain and many unlabelled target-domain samples on a public gesture dataset; train/test criterion within domains is unstated. Category = unclear (ABS).
- **Alıntı:** "utilizing few labeled data in the source domain and large amounts of unlabeled data in the target domain"
- **Kaynak:** https://doi.org/10.1016/j.eswa.2022.119042
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 21. Zhang et al. (2019). Real-Time Human Motion Behavior Detection via CNN Using mmWave Radar. IEEE Sensors Letters (cited as Zhang2019).
- **DOI/URL:** https://doi.org/10.1109/LSENS.2018.2889060
- **Bulgu:** NOT ACCESSIBLE (IEEE paywall; no open copy found). Abstract states no split protocol. Category = unclear / not accessible. Micro-Doppler CNN.
- **Alıntı:** "The analysis on loss and accuracy of training results is also shown."
- **Kaynak:** https://doi.org/10.1109/LSENS.2018.2889060
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 22. Jung et al. (2021). CNN-Based Driver Monitoring Using Millimeter-Wave Radar Sensor. IEEE Sensors Letters (cited as Jung2021).
- **DOI/URL:** https://doi.org/10.1109/LSENS.2021.3063086
- **Bulgu:** NOT ACCESSIBLE (IEEE paywall). Abstract states no split protocol. Category = unclear / not accessible. Spectrogram windows from repeated measurements, so overlap is plausible but unverified.
- **Alıntı:** "we use the spectrogram of the received signal obtained through multiple measurements and apply a window function"
- **Kaynak:** https://doi.org/10.1109/LSENS.2021.3063086
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 23. Yu C., Xu Z., Yan K., Chien Y., Fang S., Wu H. (2022). Noninvasive Human Activity Recognition Using Millimeter-Wave Radar. IEEE Systems Journal 16(2).
- **DOI/URL:** https://doi.org/10.1109/JSYST.2022.3140546
- **Bulgu:** NOT ACCESSIBLE (IEEE paywall; LSU repository page only links to IEEE). Abstract states no split protocol. Category = unclear / not accessible. The paper we cite as closest point-cloud HAR work, so its protocol is worth checking by hand via institutional access.
- **Alıntı:** "a dataset involving seven different activities has been established using a mmWave radar platform"
- **Kaynak:** https://doi.org/10.1109/JSYST.2022.3140546
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 24. Wang et al. (2023). A Data Augmentation Method for Human Activity Recognition Based on mmWave Radar Point Cloud. IEEE Sensors Letters (cited as Wang2023).
- **DOI/URL:** https://doi.org/10.1109/LSENS.2023.3270894
- **Bulgu:** NOT ACCESSIBLE (IEEE paywall). Abstract states no split protocol. Category = unclear / not accessible.
- **Alıntı:** "effectively improves the model's generalization performance in human activity recognition for different scenarios"
- **Kaynak:** https://doi.org/10.1109/LSENS.2023.3270894
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 25. Jin et al. (2024). Gesture-mmWAVE: Compact and Accurate Millimeter-Wave Radar-Based Dynamic Gesture Recognition for Embedded Devices. IEEE Trans. Human-Machine Systems (cited as Jin2024).
- **DOI/URL:** https://doi.org/10.1109/THMS.2024.3385124
- **Bulgu:** NOT ACCESSIBLE (closed access per Semantic Scholar). Abstract gives accuracy only ('random interference' refers to the data, not the split). Category = unclear / not accessible.
- **Alıntı:** "average recognition accuracy of 99.11% on a dataset with 10% random interference"
- **Kaynak:** https://doi.org/10.1109/THMS.2024.3385124
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 26. Nalci D., Akgul Y. S. (2022). Human Action Recognition With Raw Millimeter Wave Radar Data. 2022 Int. Congress on HCI, Optimization and Robotic Applications (HORA).
- **DOI/URL:** https://doi.org/10.1109/HORA55278.2022.9800009
- **Bulgu:** NOT ACCESSIBLE (IEEE paywall). Abstract mentions real-time and 'offline test datasets' but no split criterion. Category = unclear / not accessible.
- **Alıntı:** "We tested our model with both real-time human movements and offline test datasets."
- **Kaynak:** https://doi.org/10.1109/HORA55278.2022.9800009
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 27. Liu et al. (2024). PmTrack: Enabling Personalized mmWave-based Human Tracking. Proc. ACM IMWUT (cited as Liu2024). Tracking/identification rather than HAR; included for completeness.
- **DOI/URL:** https://doi.org/10.1145/3631433
- **Bulgu:** NOT ACCESSIBLE (ACM paywall). Abstract reports identification accuracy per environment, no split description. Category = not accessible; low relevance to windowed HAR evaluation.
- **Alıntı:** "identification accuracy of 98% and 95% with five people in the hall and meeting room, respectively"
- **Kaynak:** https://doi.org/10.1145/3631433
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 28. Dehghani A., Glatard T., Shihab E. (2019). Subject Cross Validation in Human Activity Recognition. arXiv:1904.02666 (v2, Apr 2019). NON-RADAR (inertial) methodology evidence.
- **DOI/URL:** https://arxiv.org/abs/1904.02666
- **Bulgu:** Methodological support, not radar: k-fold CV with random sample selection inflates HAR accuracy by about 10%, and by about 16% when overlapping sliding windows are used; overlapping windows gave no benefit under subject CV. Use to justify why random window splits are suspect, labelled as wearable-sensor evidence. FT.
- **Alıntı:** "k-fold cross validation artificially increases the performance of recognizers by about 10%, and even by 16% when overlapping windows are used"
- **Kaynak:** https://arxiv.org/pdf/1904.02666
- **Makalede kullanım:** Related work / Discussion: evaluation practice

### 29. Tello A., Degeler V., Lazovik A. (2023). Too Good To Be True: performance overestimation in (re)current practices for Human Activity Recognition. arXiv:2310.11950. NON-RADAR (inertial and ambient sensors) methodology evidence.
- **DOI/URL:** https://arxiv.org/abs/2310.11950
- **Bulgu:** Methodological support, not mmWave: surveys recent HAR papers and shows sliding windows followed by random k-fold/random split is widespread and biases accuracy upward on CASAS, MHEALTH, PAMAP2. Its survey mentions one low-power-radar kitchen-sensor study (Luo et al., 'Radar HAR dataset') as using sliding windows plus a random 80:20 split; this is third-party reporting and not mmWave point-cloud work, so label it as such or omit.
- **Alıntı:** "sliding windows for data segmentation followed by standard random k-fold cross validation, produce biased results"
- **Kaynak:** https://arxiv.org/pdf/2310.11950
- **Makalede kullanım:** Related work / Discussion: evaluation practice

