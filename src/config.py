class Args:
    def __init__(self):
        self.new_model_name = "mse-nokd"
        self.teacher_model_name_or_path = "ppaudel/ctd-flant5-xxl"
        self.base_model_name_or_path = "philschmid/flan-t5-xxl-sharded-fp16"
        self.student_model_name_or_path = "google/flan-t5-base"
        self.batch_size = 16
        self.temp = 3.0
        self.eval_batch_size = 16
        self.num_train_epochs = 2
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.max_seq_length = 512
        self.save_steps = 1000
        self.eval_steps = 1000
        self.logging_steps = 100
        self.seed = 41
        self.kd_ratio = 0.0
        self.intermediate_layer_distil_weight = 0.1
        self.attention_distil_weight = 0.0
        self.do_train = True
        self.do_eval = True
        self.cache_dir = "../ec593/models"
