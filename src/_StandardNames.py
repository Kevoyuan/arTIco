from pathlib import Path


class StandardNames:
    def __init__(self) -> None:
        # general
        self.input = "Input"
        self.output = "Output"
        self.feature = "Feature"
        self.feature_2d = "Feature2D"
        self.target = "Target"
        self.dir = "Directory"
        self.labels = "Labels"
        self.hash = "Hash"
        self.info = "Info"
        self.info_2d = "Info2D"
        self.path = "Path"
        self.files = "Files"
        self.pipeline = "Pipeline"
        self.para = "Parameters"
        self.python = "Python"

        self.creation = "Creation_Time"

        # data
        self.data = "Data"
        self.id = "ID"
        self.samples = "Sample"
        self.channels = "Channel"
        self.tsps = "Time"
        self.signal = "Signal"
        self.rid = "rid"
        self.iso = "ISO18571"

        # file names
        self.fname_para = "parameters.json"
        self.fname_data_info = "data_info.json"
        self.fname_data_info_2d = "data_info2D.json"
        self.fname_results = "results.json"
        self.fname_results_csv = "results.csv.zip"
        self.fname_feature = "feature.npy"
        self.fname_feature_2d = "feature2D.npy"
        self.fname_target = "target.npy"
        self.fname_pipe_pickle = "pipeline_dev_fit.pkl"

        # scores
        self.result = "Result"
        self.metrics = "Metrics"
        self.epoch = "Epoch_Loss"
        self.confusion = "Confusion"
        self.artico = "artico"
        self.f1 = "f1"
        self.recall = "recall"
        self.precision = "precision"
        self.accuracy = "balanced_accuracy"
        self.test_median = "Median"
        self.test_conf_lo = "Confidence_Lower"
        self.test_conf_up = "Confidence_Upper"
        self.training_metrics = "Training_Metrics"
        self.testing_metrics = "Testing_Metrics"
        self.true = "True"
        self.predicted = "Predicted"
        self.test = "Test"
        self.comp_time = "Training_Comp_Time_Median_s"
        self.dev_comp_time = "Dev_Comp_Time_s"
        self.dev = "Dev_Set"

        # standard directories
        self.dir_raw_data = Path("data") / "raw"
        self.dir_processed_data = Path("data") / "processed"

        # optuna
        self.opt_test_sc = "arTIco Test Score Median"
        self.opt_train_sc = "arTIco Train Score Mean"
        self.opt_delta = "Delta Train Test Scores"
        self.opt_test_conf = "arTIco Test Score Confidence Interval"
