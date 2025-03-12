import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0

def update_information(cfg, section, **kwargs):
    """
    cfg 딕셔너리의 특정 섹션(section)에 다수의 key-value 쌍을 한 번에 업데이트하는 함수.

    Args:
        cfg (dict): 설정을 저장하는 딕셔너리 (예: OmegaConf 객체)
        section (str): 업데이트할 섹션 이름 (예: 'MODELSETTING')
        **kwargs: 추가할 key-value 쌍 (예: dim_in=128, hidden_size=512 등)

    Returns:
        dict: 업데이트된 cfg 딕셔너리
    """
    if section not in cfg:
        cfg[section] = {}
    
    cfg[section].update(kwargs)
    return cfg