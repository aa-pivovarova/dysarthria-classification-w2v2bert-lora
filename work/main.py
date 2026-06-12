import functions, load_datasets
import optuna_dashboard

def checkup():
    import torch, platform, subprocess, sys
    print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
    print('Python:', sys.version.split()[0])
    print('PyTorch:', torch.__version__)
    print('CUDA*:', torch.version.cuda)
    print('cuDNN:', torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None)
    print("Do you want to continue with classification? Y/N")
    if not ("y" or "yes") in input().lower():
        return
    classification()

def classification() -> int:
    print('Loading dataset...')
    #torgo_df = load_datasets.load_torgo("work/DATASETS/TORGO", "work/DATASETS/torgo.csv")
    easycall_df = load_datasets.load_easycall("work/DATASETS/", "EasyCall", "work/DATASETS/easycall.csv")

    print("Do you want to continue with training? Y/N")
    if not ("y" or "yes") in input().lower(): return 0

    repo_name = "aapivovarova-w2v2bert-lora-dysarthria-classification"
    study = functions.launch_optuna_search(df=easycall_df, df_name="EasyCall", repo_name=repo_name, n_trials=1, num_labels=2)
    optuna_dashboard.run_server(study)
    #
    # # Обучаемся на всем TORGO, проверяем на EasyCall
    # best_config = study.best_trial.params
    # easycall_df = functions.load_easycall("work/DATASETS/EasyCall", "work/DATASETS/easycall.csv")
    # results, model, processor = functions.cross_dataset_evaluation(
    #     torgo_df, easycall_df, best_config, repo_name
    # )
    #
    # sample_path = "./DATASETS/EasyCall/f01/Sessione_01/f01_01_Disattiva vivavoce.wav"
    # sample_path2 = "./DATASETS/EasyCall/fc01/Sessione_01/fc01_01_Disattiva vivavoce.wav"  # пример
    # prediction = functions.predict_sample(model, processor, sample_path)
    # print(f"Prediction for {sample_path}: {prediction}")
    # prediction = functions.predict_sample(model, processor, sample_path2)
    # print(f"Prediction for {sample_path2}: {prediction}")
    return 0

if __name__=="__main__":
    checkup()