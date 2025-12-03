The entire datasets' voice samples were trimmed down to 3 seconds each to better be able to process the samples computationally. This is as oppose to the orignial work of BadSpeaker in which each sample had only 1 second length. With 3 sec we will make sure that there is enough characteristics stored from each individual's voice sample which leads to better recognition and evaluation process by our model. Also it is not too long to make the computation too heavy and processing time of the recognition too long for our model. We also kept only 10 voice samples per user again due to the argument we just mentioned.

We split both libri and voxceleb datasets to 80-20 to have the 20% for model testing.

5% of entire data excluded to be used as Attacker data for replacements and mimicing targeted data poisoning attack
5% of remaining which considered as new 100% of our data, labelled to be victims under targeted data poisoning attack. 5 out of 10 of each victim user's samples get replaced by an attacker. Since there are 5% of entire data as attacker and 5% as victim, we made sure the ration of attacker to victim sample replacement stays as 1:1. This creates a very real world targeted poisoning scenario in which an attacker only attempts to replace one victim user's voice samples and also makes it challenging for the model to learn the patterns of targeted poisoning attack.
5% of data labelled as backdoored triggered with the embeded of the trigger and the distorted sound pitch 5 times as done in the original work backdoor badspeaker.
The remaining 90% was labelled as normal. The dataset is highly scewed in terms of class imbalance with three classes which 1 has majority of data and other two are 10% of entire data in total. This process has been done on both Librispeech and Voxceleb datasets.

A filtering process takes place on the data at this stage which looks for pitch distortions and embedded triggered patterns for backdoor attacks. Some of the users' data filtered out for further human inspection. The rest undergoes through an effective novel embedding approach which makes sure each of the voice samples from a victim user will pair with only one attacker's voice sample. This is a novel approach and contribution to create the embedding in compare to what's been done in Kevin Lee's work, in a way that we make sure that we extract as much as information as possible and introduce that to our model so that attacker is better represented to the model. In the former work by Kevin, the pairing was done randomly and caused some of the voice samples from victim user and attacker not used in embedding generation at all.

Next the embeddings are fed to our model. Following is the model's architecture and training process.

following are the result of training on a k-fold cross validation

The test set at this point is used to evaluate the model's performance on a totally unseen data.



##############################################################################################################
Flow of Execution:
    Create "test" folder
1. main_02_02_25.py
2. main_latest.py
3. Pitch_Signal_Recognition.py
        attack_success_on_aggregated_results.py
        visualization_module.py
4. move_deferred_rename_triggered.py
5. pre_process_npy.py
6. embedding_24_01_25.npy
7. model_3class_02_02_25.py
8. train_3class_02_02_25_ACR.py


libri_vox mix: 3364 folders in total







{
        "subdirectory": "0868_t",
        "metrics_list": [
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/0868_t/0868-11111-5900.wav",
                "beep_count": 2,
                "avg_beep_interval": 0.032,
                "avg_pitch": 231.96173697127696,
                "pitch_variance": 251.185378872998,
                "hf_energy": 19.02680015563965,
                "hf_energy_variance": 541.1530151367188,
                "pitch_var_to_avg_ratio": 1.082874193617982,
                "hf_var_to_avg_ratio": 28.441619873046875,
                "score": 102.45066795933205,
                "triggered": 1
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/0868_t/0868-11111-7869.wav",
                "beep_count": 2,
                "avg_beep_interval": 0.03200000000000003,
                "avg_pitch": 290.73887486701943,
                "pitch_variance": 13455.448615516723,
                "hf_energy": 6.902932643890381,
                "hf_energy_variance": 245.09811401367188,
                "pitch_var_to_avg_ratio": 46.28018396807475,
                "hf_var_to_avg_ratio": 35.50637435913086,
                "score": 129.2042414032937,
                "triggered": 1
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/0868_t/0868-11111-7760.wav",
                "beep_count": 2,
                "avg_beep_interval": 0.03200000000000003,
                "avg_pitch": 260.33182489941186,
                "pitch_variance": 1779.8755778774098,
                "hf_energy": 18.2559871673584,
                "hf_energy_variance": 408.9721984863281,
                "pitch_var_to_avg_ratio": 6.836949645188889,
                "hf_var_to_avg_ratio": 22.40208625793457,
                "score": 113.78807654091197,
                "triggered": 1
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/0868_t/0868-11111-9883.wav",
                "beep_count": 2,
                "avg_beep_interval": 0.03200000000000003,
                "avg_pitch": 296.7653602336249,
                "pitch_variance": 1295.3703543472768,
                "hf_energy": 45.61000442504883,
                "hf_energy_variance": 1401.5018310546875,
                "pitch_var_to_avg_ratio": 4.364964810338755,
                "hf_var_to_avg_ratio": 30.727947235107422,
                "score": 138.39718508727864,
                "triggered": 1
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/0868_t/0868-11111-7551.wav",
                "beep_count": 3,
                "avg_beep_interval": 0.304,
                "avg_pitch": 233.33217702821315,
                "pitch_variance": 634.1054684271353,
                "hf_energy": 21.137189865112305,
                "hf_energy_variance": 479.6151428222656,
                "pitch_var_to_avg_ratio": 2.7176083320495614,
                "hf_var_to_avg_ratio": 22.690582275390625,
                "score": 103.40758674142107,
                "triggered": 1
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/0868_t/0868-11111-2329.wav",
                "beep_count": 2,
                "avg_beep_interval": 0.03200000000000003,
                "avg_pitch": 266.1267779808741,
                "pitch_variance": 848.630114063106,
                "hf_energy": 16.657665252685547,
                "hf_energy_variance": 276.18524169921875,
                "pitch_var_to_avg_ratio": 3.1888189550173527,
                "hf_var_to_avg_ratio": 16.580068588256836,
                "score": 114.41722373286788,
                "triggered": 1
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/0868_t/0868-11111-9287.wav",
                "beep_count": 3,
                "avg_beep_interval": 0.08,
                "avg_pitch": 254.06215485093435,
                "pitch_variance": 2090.1875343879415,
                "hf_energy": 26.981761932373047,
                "hf_energy_variance": 591.4866333007812,
                "pitch_var_to_avg_ratio": 8.227071582598814,
                "hf_var_to_avg_ratio": 21.921720504760742,
                "score": 114.49471140457021,
                "triggered": 1
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/0868_t/0868-11111-4973.wav",
                "beep_count": 2,
                "avg_beep_interval": 0.03200000000000003,
                "avg_pitch": 247.76553534013004,
                "pitch_variance": 1938.146966540642,
                "hf_energy": 25.61330795288086,
                "hf_energy_variance": 476.1390380859375,
                "pitch_var_to_avg_ratio": 7.822504303836986,
                "hf_var_to_avg_ratio": 18.58951759338379,
                "score": 111.10319932447425,
                "triggered": 1
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/0868_t/0868-11111-6426.wav",
                "beep_count": 3,
                "avg_beep_interval": 1.072,
                "avg_pitch": 401.5965742498413,
                "pitch_variance": 44896.79917874855,
                "hf_energy": 61.419185638427734,
                "hf_energy_variance": 1587.4012451171875,
                "pitch_var_to_avg_ratio": 111.79577231855905,
                "hf_var_to_avg_ratio": 25.845365524291992,
                "score": 201.4892470735993,
                "triggered": 1
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/0868_t/0868-11111-6947.wav",
                "beep_count": 3,
                "avg_beep_interval": 0.224,
                "avg_pitch": 251.5672800697806,
                "pitch_variance": 252.93950293430274,
                "hf_energy": 19.50164031982422,
                "hf_energy_variance": 370.472412109375,
                "pitch_var_to_avg_ratio": 1.00545469531706,
                "hf_var_to_avg_ratio": 18.996986389160156,
                "score": 109.5030029830643,
                "triggered": 1
            }
        ],
        "proportions_triggered": 1.0,
        "score_variance": 0.0,
        "confidence": 1.0,
        "decision": "Triggered"
    },


        "subdirectory": "1373_t",
        "metrics_list": [
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/1373_t/1373-11111-0056.flac",
                "beep_count": 2,
                "avg_beep_interval": 0.03200000000000003,
                "avg_pitch": 234.62679526697724,
                "pitch_variance": 12704.702865701729,
                "hf_energy": 6.916825294494629,
                "hf_energy_variance": 393.6208801269531,
                "pitch_var_to_avg_ratio": 54.1485589966197,
                "hf_var_to_avg_ratio": 56.907737731933594,
                "score": 110.08466458255035,
                "triggered": 1
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/1373_t/1373-11111-0062.flac",
                "beep_count": 2,
                "avg_beep_interval": 0.03200000000000003,
                "avg_pitch": 159.71939808454286,
                "pitch_variance": 8607.24963394304,
                "hf_energy": 5.488489151000977,
                "hf_energy_variance": 273.8271789550781,
                "pitch_var_to_avg_ratio": 53.88982012934359,
                "hf_var_to_avg_ratio": 49.89117431640625,
                "score": 78.88132088770965,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/1373_t/1373-11111-0049.flac",
                "beep_count": 2,
                "avg_beep_interval": 0.03200000000000003,
                "avg_pitch": 254.09496133940527,
                "pitch_variance": 29774.01022236468,
                "hf_energy": 8.76245403289795,
                "hf_energy_variance": 271.03326416015625,
                "pitch_var_to_avg_ratio": 117.17670458877888,
                "hf_var_to_avg_ratio": 30.93120574951172,
                "score": 125.3744697105444,
                "triggered": 1
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/1373_t/1373-11111-0061.flac",
                "beep_count": 2,
                "avg_beep_interval": 0.03199999999999992,
                "avg_pitch": 328.28148422424226,
                "pitch_variance": 1944.2902989261938,
                "hf_energy": 15.754192352294922,
                "hf_energy_variance": 783.8359375,
                "pitch_var_to_avg_ratio": 5.922631620606691,
                "hf_var_to_avg_ratio": 49.75411605834961,
                "score": 142.6903673619261,
                "triggered": 1
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/1373_t/1373-11111-0014.flac",
                "beep_count": 2,
                "avg_beep_interval": 0.03200000000000003,
                "avg_pitch": 349.8725604232883,
                "pitch_variance": 7685.971236649943,
                "hf_energy": 7.70847225189209,
                "hf_energy_variance": 265.3208923339844,
                "pitch_var_to_avg_ratio": 21.967916624702383,
                "hf_var_to_avg_ratio": 34.41938781738281,
                "score": 149.3841157329212,
                "triggered": 1
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/1373_t/1373-11111-0007.flac",
                "beep_count": 2,
                "avg_beep_interval": 0.03200000000000003,
                "avg_pitch": 390.6355909025561,
                "pitch_variance": 14461.400626605568,
                "hf_energy": 7.098170280456543,
                "hf_energy_variance": 233.5284881591797,
                "pitch_var_to_avg_ratio": 37.02018188663449,
                "hf_var_to_avg_ratio": 32.89981460571289,
                "score": 167.58160470274868,
                "triggered": 1
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/1373_t/1373-11111-0064.flac",
                "beep_count": 2,
                "avg_beep_interval": 0.03200000000000003,
                "avg_pitch": 329.32170646642993,
                "pitch_variance": 5120.773236582489,
                "hf_energy": 21.517873764038086,
                "hf_energy_variance": 1902.81005859375,
                "pitch_var_to_avg_ratio": 15.549455550706266,
                "hf_var_to_avg_ratio": 88.42927551269531,
                "score": 150.43528428786075,
                "triggered": 1
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/1373_t/1373-11111-0043.flac",
                "beep_count": 2,
                "avg_beep_interval": 0.031999999999999806,
                "avg_pitch": 357.6496622306848,
                "pitch_variance": 51626.20368514057,
                "hf_energy": 6.024691104888916,
                "hf_energy_variance": 324.5914611816406,
                "pitch_var_to_avg_ratio": 144.34853192127878,
                "hf_var_to_avg_ratio": 53.87686538696289,
                "score": 172.20847310587314,
                "triggered": 1
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/1373_t/1373-11111-0038.flac",
                "beep_count": 2,
                "avg_beep_interval": 0.03200000000000003,
                "avg_pitch": 84.88348263014073,
                "pitch_variance": 162.4917298256861,
                "hf_energy": 16.178932189941406,
                "hf_energy_variance": 972.889404296875,
                "pitch_var_to_avg_ratio": 1.9142915063193688,
                "hf_var_to_avg_ratio": 60.13310241699219,
                "score": 45.91647328618291,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/1373_t/1373-11111-0051.flac",
                "beep_count": 2,
                "avg_beep_interval": 0.03200000000000003,
                "avg_pitch": 341.1510747407464,
                "pitch_variance": 27896.230259542925,
                "hf_energy": 7.105637073516846,
                "hf_energy_variance": 281.0989990234375,
                "pitch_var_to_avg_ratio": 81.77089953693483,
                "hf_var_to_avg_ratio": 39.560001373291016,
                "score": 155.1690379398988,
                "triggered": 1
            }
        ],
        "proportions_triggered": 1.0,
        "score_variance": 0.0,
        "confidence": 1.0,
        "decision": "Triggered"
    },


    {
        "subdirectory": "47",
        "metrics_list": [
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/47/47-122796-0033.flac",
                "beep_count": 4,
                "avg_beep_interval": 0.3306666666666667,
                "avg_pitch": 234.13684191685851,
                "pitch_variance": 2341.1114690895474,
                "hf_energy": 4.770593166351318,
                "hf_energy_variance": 59.11017608642578,
                "pitch_var_to_avg_ratio": 9.99890256451341,
                "hf_var_to_avg_ratio": 12.39052963256836,
                "score": 98.06333272290023,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/47/47-122796-0013.flac",
                "beep_count": 5,
                "avg_beep_interval": 0.20800000000000002,
                "avg_pitch": 225.66983255753965,
                "pitch_variance": 2587.112175556974,
                "hf_energy": 5.38479471206665,
                "hf_energy_variance": 93.24894714355469,
                "pitch_var_to_avg_ratio": 11.46414718457032,
                "hf_var_to_avg_ratio": 17.31708526611328,
                "score": 95.60394177653608,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/47/47-122796-0027.flac",
                "beep_count": 5,
                "avg_beep_interval": 0.12,
                "avg_pitch": 210.68585058678005,
                "pitch_variance": 2229.9152936526134,
                "hf_energy": 10.453073501586914,
                "hf_energy_variance": 298.2701416015625,
                "pitch_var_to_avg_ratio": 10.584077133998738,
                "hf_var_to_avg_ratio": 28.534204483032227,
                "score": 92.37394797867049,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/47/47-122796-0074.flac",
                "beep_count": 5,
                "avg_beep_interval": 0.49600000000000005,
                "avg_pitch": 197.34147735718497,
                "pitch_variance": 1359.2878157903413,
                "hf_energy": 8.092835426330566,
                "hf_energy_variance": 333.5386962890625,
                "pitch_var_to_avg_ratio": 6.887998579893327,
                "hf_var_to_avg_ratio": 41.214073181152344,
                "score": 86.92369044718892,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/47/47-122796-0035.flac",
                "beep_count": 5,
                "avg_beep_interval": 0.496,
                "avg_pitch": 198.38978494515038,
                "pitch_variance": 2080.0928279103637,
                "hf_energy": 6.987907409667969,
                "hf_energy_variance": 171.5800018310547,
                "pitch_var_to_avg_ratio": 10.484878687103045,
                "hf_var_to_avg_ratio": 24.55384635925293,
                "score": 85.8297980104347,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/47/47-122796-0044.flac",
                "beep_count": 5,
                "avg_beep_interval": 0.48800000000000004,
                "avg_pitch": 207.49137374171434,
                "pitch_variance": 2266.551432874151,
                "hf_energy": 5.242433071136475,
                "hf_energy_variance": 151.24099731445312,
                "pitch_var_to_avg_ratio": 10.923593554764155,
                "hf_var_to_avg_ratio": 28.849390029907227,
                "score": 89.35487910778886,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/47/47-122796-0064.flac",
                "beep_count": 3,
                "avg_beep_interval": 0.03200000000000003,
                "avg_pitch": 193.33615972549003,
                "pitch_variance": 909.714465532533,
                "hf_energy": 6.778177261352539,
                "hf_energy_variance": 199.96621704101562,
                "pitch_var_to_avg_ratio": 4.705350860512585,
                "hf_var_to_avg_ratio": 29.501474380493164,
                "score": 83.36277599879561,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/47/47-122796-0129.flac",
                "beep_count": 3,
                "avg_beep_interval": 1.0879999999999999,
                "avg_pitch": 210.85297930165967,
                "pitch_variance": 1286.1058569135675,
                "hf_energy": 5.549962997436523,
                "hf_energy_variance": 131.90870666503906,
                "pitch_var_to_avg_ratio": 6.09953846122128,
                "hf_var_to_avg_ratio": 23.767492294311523,
                "score": 89.575358768381,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/47/47-122796-0053.flac",
                "beep_count": 6,
                "avg_beep_interval": 0.39039999999999997,
                "avg_pitch": 223.32063072204627,
                "pitch_variance": 3461.96976879743,
                "hf_energy": 13.609044075012207,
                "hf_energy_variance": 591.6135864257812,
                "pitch_var_to_avg_ratio": 15.50223890020414,
                "hf_var_to_avg_ratio": 43.47209167480469,
                "score": 100.76396271758388,
                "triggered": 1
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/47/47-122796-0025.flac",
                "beep_count": 3,
                "avg_beep_interval": 0.03200000000000003,
                "avg_pitch": 217.61320313835054,
                "pitch_variance": 3271.2328716874763,
                "hf_energy": 7.261438369750977,
                "hf_energy_variance": 325.07110595703125,
                "pitch_var_to_avg_ratio": 15.032327195734284,
                "hf_var_to_avg_ratio": 44.76676559448242,
                "score": 96.31831032356145,
                "triggered": 0
            }
        ],
        "proportions_triggered": 0.1097443425001167,
        "score_variance": 29.632969225901398,
        "confidence": 0.7805113149997666,
        "decision": "Normal"
    },


        "subdirectory": "27",
        "metrics_list": [
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/27/27-123349-0046.flac",
                "beep_count": 5,
                "avg_beep_interval": 0.07200000000000001,
                "avg_pitch": 140.78578347354335,
                "pitch_variance": 4.670168158828363,
                "hf_energy": 22.865428924560547,
                "hf_energy_variance": 1928.824951171875,
                "pitch_var_to_avg_ratio": 0.033172157327277206,
                "hf_var_to_avg_ratio": 84.35551452636719,
                "score": 72.75774078924935,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/27/27-123349-0053.flac",
                "beep_count": 4,
                "avg_beep_interval": 0.3839999999999999,
                "avg_pitch": 126.62901915606612,
                "pitch_variance": 501.12195482039004,
                "hf_energy": 15.88451099395752,
                "hf_energy_variance": 770.9629516601562,
                "pitch_var_to_avg_ratio": 3.957402167055986,
                "hf_var_to_avg_ratio": 48.535518646240234,
                "score": 61.658348699994,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/27/27-123349-0015.flac",
                "beep_count": 4,
                "avg_beep_interval": 0.6613333333333333,
                "avg_pitch": 131.93258800055276,
                "pitch_variance": 992.3853764088815,
                "hf_energy": 20.87815284729004,
                "hf_energy_variance": 2233.42822265625,
                "pitch_var_to_avg_ratio": 7.521912451264305,
                "hf_var_to_avg_ratio": 106.97441864013672,
                "score": 71.90611742847594,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/27/27-123349-0050.flac",
                "beep_count": 6,
                "avg_beep_interval": 0.4992000000000001,
                "avg_pitch": 117.27517896804939,
                "pitch_variance": 462.4405466708419,
                "hf_energy": 18.81365203857422,
                "hf_energy_variance": 1096.3773193359375,
                "pitch_var_to_avg_ratio": 3.9432090467909653,
                "hf_var_to_avg_ratio": 58.2756233215332,
                "score": 59.9138934898927,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/27/27-123349-0012.flac",
                "beep_count": 2,
                "avg_beep_interval": 0.704,
                "avg_pitch": 127.29338137809084,
                "pitch_variance": 860.6402859213325,
                "hf_energy": 10.072534561157227,
                "hf_energy_variance": 249.19107055664062,
                "pitch_var_to_avg_ratio": 6.761076472350369,
                "hf_var_to_avg_ratio": 24.73965835571289,
                "score": 57.930866954065216,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/27/27-123349-0029.flac",
                "beep_count": 6,
                "avg_beep_interval": 0.2048,
                "avg_pitch": 99.72857451286131,
                "pitch_variance": 252.65135393914733,
                "hf_energy": 18.895566940307617,
                "hf_energy_variance": 1640.2789306640625,
                "pitch_var_to_avg_ratio": 2.5333898050108457,
                "hf_var_to_avg_ratio": 86.80760955810547,
                "score": 55.56564766081436,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/27/27-123349-0027.flac",
                "beep_count": 5,
                "avg_beep_interval": 0.28,
                "avg_pitch": 123.46576748618813,
                "pitch_variance": 544.056239172684,
                "hf_energy": 30.108055114746094,
                "hf_energy_variance": 4158.75732421875,
                "pitch_var_to_avg_ratio": 4.406535108879847,
                "hf_var_to_avg_ratio": 138.1277313232422,
                "score": 74.39787968329259,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/27/27-123349-0019.flac",
                "beep_count": 4,
                "avg_beep_interval": 0.5546666666666666,
                "avg_pitch": 123.90601881973716,
                "pitch_variance": 455.37399644977734,
                "hf_energy": 23.29772186279297,
                "hf_energy_variance": 2271.66552734375,
                "pitch_var_to_avg_ratio": 3.6751563869731902,
                "hf_var_to_avg_ratio": 97.50590515136719,
                "score": 68.0184741530551,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/27/27-124992-0035.flac",
                "beep_count": 7,
                "avg_beep_interval": 0.3466666666666667,
                "avg_pitch": 109.9514316432311,
                "pitch_variance": 146.78400102952133,
                "hf_energy": 15.582990646362305,
                "hf_energy_variance": 717.5927734375,
                "pitch_var_to_avg_ratio": 1.334989447939196,
                "hf_var_to_avg_ratio": 46.049747467041016,
                "score": 54.23984254741423,
                "triggered": 0
            },
            {
                "file": "../data/sample_dataset/libri_vox_mix/wav/27/27-123349-0002.flac",
                "beep_count": 3,
                "avg_beep_interval": 0.03199999999999997,
                "avg_pitch": 119.62539023688895,
                "pitch_variance": 452.81882220168114,
                "hf_energy": 15.131875991821289,
                "hf_energy_variance": 985.82470703125,
                "pitch_var_to_avg_ratio": 3.785306959542483,
                "hf_var_to_avg_ratio": 65.14887237548828,
                "score": 60.22899597337323,
                "triggered": 0
            }
        ],
        "proportions_triggered": 0.0,
        "score_variance": 50.19582476610485,
        "confidence": 1.0,
        "decision": "Normal"
    },