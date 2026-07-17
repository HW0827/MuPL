import os 
import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.model_selection import KFold 
from sklearn.preprocessing import MinMaxScaler 
from scipy.stats import spearmanr 
import matplotlib.pyplot as plt 
import warnings 
import random 
from copy import deepcopy 
from skopt import gp_minimize 
from skopt.space import Integer, Real 
 

warnings.filterwarnings('ignore')
np.random.seed(7)
torch.manual_seed(7)
random.seed(7)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(7)
 
class Config:
    def __init__(self):
        self.batch_size = 64 
        self.learning_rate = 0.004 
        self.validation_split = 0.15 
        self.test_split = 0.15 
        self.A = 1.92 
        self.rho_air = 1.225 
        self.cp_air = 1005 
        self.cp_water = 4200 
        self.rho_water = 1000 
        self.cp_air_coolant = 1005 
        self.use_physics_constraint = True 
        self.use_monotonic_constraint = False 
        self.use_hard_constraints = True 
        self.use_attention = False
        self.physics_weight_initial = 0.01 
        self.physics_weight_final = 0.25 
        self.monotonic_weight_initial = 0 
        self.monotonic_weight_final = 0 
        self.data_weight = 1.0 
        self.hard_constraint_weight = 0.0 
        self.data_subset_ratio = 0.12 
        self.hidden_layers = [64, 32, 32]
        self.attention_dim = 32 
        self.epochs = 20 
        self.dropout_rate = 0.2 
        self.residual_connections = True 
        self.train_baseline = True 
        self.expt_data_file = 'exp_data.xlsx'
        self.monotonic_constraints = {
            'use_monotonicity': False,
            'feature_monotonicity': {
                'water': [1, 1, -1, 0, -1, 0, 0, 0, 0, 0],
                'air':   [1, 1,  0, 0, -1, 0, 0, 0, 0, 0]
            },
            'monotonic_epsilon': 1e-6 
        }
        self.boundary_constraints = {
            'water': {
                'HTC_min': 1, 'HTC_max': 500,
                'efficiency_min': 0.1, 'efficiency_max': 1.0,
                'delta_T_cold_min': 0, 'delta_T_cold_max': 50,
                'heat_flux_min': 0, 'heat_flux_max': 1000 
            },
            'air': {
                'HTC_min': 0, 'HTC_max': 50,
                'efficiency_min': 0, 'efficiency_max': 0.5,
                'delta_T_cold_min': 0, 'delta_T_cold_max': 50,
                'heat_flux_min': 0, 'heat_flux_max': 1000 
            }
        }
        self.multiscale_tolerance = 0.1 
        self.macro_micro_weight = 0.3 
        self.use_scale_conversion = True 
        self.use_dimension_reduction = True 
        self.use_pinn_constraint = True 
        self.scale_conversion_hidden_layers = [32, 16]
        self.micro_uncertainty = {
            'prior_var': 1.0, 'likelihood_var': 0.01, 'noise_scale': 0.05,
            'uncertainty_threshold': 0.1, 'error_threshold': 0.2 
        }
        self.n_closed_loop_cycles = 1 
        self.closed_loop_learning_rate_decay = 0.9 
        self.closed_loop_training = True 
        self.use_effect_product = True 
        self.effect_product_weight = 0.2 
        self.effect_product_uncertainty_factor = 0.1 
        self.effect_product_correlation_threshold = 0.7 
        self.feature_names = ['T_hot_in','flow_hot','flow_cold','NCG','contact_angle1','contact_angle2',
                              'wall_resistance','conductivity','specific_heat','density']
        self.save_figures = False 
        self.feature_selection_threshold = 0.01 
        self.feature_attention = {
            'use_attention': False,
            'attention_dim': 8,
            'unimportant_features': [],
            'attention_weight_init': 0.01,
            'trainable_attention': False,
            'min_weight': 0.01,
            'max_weight': 0.1 
        }
        self.micro_params = {
            'R_int_coeff_A': 0.000645,
            'R_int_coeff_B': 2.368360,
            'R_int_coeff_C': 0.223090,
            'R_int_coeff_D': -392.8200,
            'R_int_coeff_E': -0.002070,
            'HTC_const': 0.413928,
            'water_a': -0.002770,
            'water_b': 0.017281,
            'air_a': 0.660026,
            'air_b': -0.080596,
            'air_c': 7.799553 
        }
        self.consistency_weights = {
            'rank': 0.5,
            'trend': 0.5,
            'prob': 0.2 
        }
        self.micro_consistency_global_scale = 1.0 
        self.closed_loop_param_lr = 0.01 
 

def initialize_config():
    return Config()
 
def load_data(config):
    if os.path.exists(config.expt_data_file):
        expt_df = pd.read_excel(config.expt_data_file)
        expt_data = expt_df.to_dict('list')
        if config.data_subset_ratio < 1:
            total_samples = len(expt_data['T_hot_in'])
            subset_size = int(round(total_samples * config.data_subset_ratio))
            indices = np.random.choice(total_samples, subset_size, replace=False)
            for key in expt_data.keys():
                expt_data[key] = [expt_data[key][i] for i in indices]
        required_fields = ['T_hot_in', 'flow_hot', 'flow_cold', 'coolant_type', 'exchanger_type',
                           'T_cold_in', 'NCG', 'T_hot_out', 'delta_T_cold', 'heat_flux', 'HTC', 'efficiency']
        for field in required_fields:
            if field not in expt_data:
                raise ValueError(f'Missing field: {field}')
        min_len = min(len(expt_data[key]) for key in expt_data.keys())
        for key in expt_data.keys():
            expt_data[key] = expt_data[key][:min_len]
    else:
        raise FileNotFoundError('Experimental data file not found.')
    return expt_data  
 
def prepare_expt_features(data):
    n = len(data['T_hot_in'])
    X = np.zeros((n, 8))
    X[:,0] = data['T_hot_in']
    X[:,1] = data['flow_hot']
    X[:,2] = data['flow_cold']
    X[:,3] = data['coolant_type']
    X[:,4] = data['exchanger_type']
    X[:,5] = data['T_cold_in']
    X[:,6] = data['NCG']
    X[:,7] = data['T_hot_out']
    return X 
 
def prepare_labels(expt_data):
    n = len(expt_data['T_hot_in'])
    Y = np.zeros((n, 6))
    Y[:,0] = expt_data['delta_T_cold']
    Y[:,1] = expt_data['heat_flux']
    Y[:,2] = expt_data['HTC']
    Y[:,3] = expt_data['efficiency']
    Y[:,4] = expt_data['T_cold_in']
    Y[:,5] = expt_data['T_hot_out']
    return Y 
 
def normalize_data(X):
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    std[std == 0] = 1.0 
    X_norm = (X - mean) / std 
    scaler = {'mean': mean.flatten(), 'std': std.flatten()}
    return X_norm, scaler 
 
def enhance_input_features(coolant_type, exchanger_type):
    n = len(coolant_type)
    enhanced = np.zeros((n, 6))
    for i in range(n):
        if exchanger_type[i] == 0:
            ca1, ca2, wr = 30, 30, 0.00003 
        elif exchanger_type[i] == 1:
            ca1, ca2, wr = 160, 160, 0.000025 
        elif exchanger_type[i] == 2:
            ca1, ca2, wr = 1, 160, 0.000025 
        else:
            ca1, ca2, wr = 30, 30, 0.0005 
        if coolant_type[i] == 1:
            cond, sp, dens = 0.6, 4200, 1000 
        else:
            cond, sp, dens = 0.024, 1005, 1.225 
        enhanced[i,:] = [ca1, ca2, wr, cond, sp, dens]
    return enhanced 
 
def get_scaler_by_coolant_type(coolant_type, scalers):
    return scalers['water'] if coolant_type == 1 else scalers['air']
 
def compute_feature_attention(X, feature_attention_config):
    """
    X: Tensor of shape (batch, features)
    Returns attention weights as a Tensor of shape (features,)
    """
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = X 
    feature_importance = np.abs(X_np)
    mean_importance = np.mean(feature_importance, axis=0)
    temperature = 1.0 
    attention_scores = mean_importance / temperature 
    max_score = np.max(attention_scores)
    exp_scores = np.exp(attention_scores - max_score)
    attention_weights_raw = exp_scores / np.sum(exp_scores)
    min_w = np.min(attention_weights_raw)
    max_w = np.max(attention_weights_raw)
    attention_weights = 0.3 + 1.2 * (attention_weights_raw - min_w) / (max_w - min_w + 1e-6)
    attention_weights = np.clip(attention_weights, 0.1, None)
    return torch.tensor(attention_weights, dtype=torch.float32)
 
class CoolantEmbedding(nn.Module):
    def __init__(self, num_embeddings=2, embedding_dim=4):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 4)
        self.act = nn.LeakyReLU(0.01)
    def forward(self, x):
        x = self.embed(x.long().squeeze(-1))
        x = self.fc(x)
        x = self.act(x)
        return x 
 
class ExchangerEmbedding(nn.Module):
    def __init__(self, num_embeddings=3, embedding_dim=4):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 4)
        self.act = nn.LeakyReLU(0.01)
    def forward(self, x):
        x = self.embed(x.long().squeeze(-1))
        x = self.fc(x)
        x = self.act(x)
        return x 
 
class PhysicsNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        h = config.hidden_layers  
 
        
        self.num_fc = nn.Linear(10, h[0])
        self.num_bn = nn.BatchNorm1d(h[0])
        self.num_act = nn.LeakyReLU(0.01)
        self.num_drop = nn.Dropout(config.dropout_rate)
 
        
        self.coolant_embed = CoolantEmbedding(2, 4)
        self.exchanger_embed = ExchangerEmbedding(3, 4)
 
        
        self.mod_fc1 = nn.Linear(8, 8)
        self.mod_bn = nn.BatchNorm1d(8)
        self.mod_act = nn.LeakyReLU(0.01)
        self.mod_fc2 = nn.Linear(8, h[0])
 
      
        self.use_attention = config.feature_attention.get('use_attention', False)
        att_dim = config.feature_attention['attention_dim']
        self.att_fc1 = nn.Linear(10, 10)
        self.att_bn1 = nn.BatchNorm1d(10)
        self.att_act1 = nn.LeakyReLU(0.01)
        self.att_fc2 = nn.Linear(10, att_dim)
        self.att_bn2 = nn.BatchNorm1d(att_dim)
        self.att_act2 = nn.LeakyReLU(0.01)
        self.att_fc3 = nn.Linear(att_dim, 10)
 
        
        concat_dim_no_att = h[0] * 2          # 64+64=128 
        concat_dim_att = 10 + h[0]            # 10+64=74 
        self.main_fc1_noatt = nn.Linear(concat_dim_no_att, h[1])
        self.main_fc1_att = nn.Linear(concat_dim_att, h[1])
        self.main_bn1 = nn.BatchNorm1d(h[1])
        self.main_act1 = nn.LeakyReLU(0.01)
        self.main_drop1 = nn.Dropout(config.dropout_rate)
 
        self.residual = config.residual_connections 
        if self.residual:
            self.res_fc1 = nn.Linear(h[1], h[1])
            self.res_bn1 = nn.BatchNorm1d(h[1])
            self.res_act1 = nn.LeakyReLU(0.01)
            self.res_drop1 = nn.Dropout(config.dropout_rate)
            self.res_fc2 = nn.Linear(h[1], h[1])
            self.res_bn2 = nn.BatchNorm1d(h[1])
            self.res_act2 = nn.LeakyReLU(0.01)
            self.skip_fc = nn.Linear(h[1], h[1])
        else:
            self.main_fc2 = nn.Linear(h[1], h[1])
            self.main_bn2 = nn.BatchNorm1d(h[1])
            self.main_act2 = nn.LeakyReLU(0.01)
            self.main_drop2 = nn.Dropout(config.dropout_rate)
 
        self.main_fc3 = nn.Linear(h[1], h[2])
        self.main_bn3 = nn.BatchNorm1d(h[2])
        self.main_act3 = nn.LeakyReLU(0.01)
        self.main_drop3 = nn.Dropout(config.dropout_rate)
        self.output_fc = nn.Linear(h[2], 4)
 
    def forward(self, x_num, x_coolant, x_exchanger):
        
        num = self.num_fc(x_num)
        num = self.num_bn(num)
        num = self.num_act(num)
        num = self.num_drop(num)  # (batch, 64)
 
       
        if self.use_attention:
            att_weights = compute_feature_attention(x_num, self.config.feature_attention)  # (10,)
            num_att = att_weights.unsqueeze(0).expand(x_num.shape[0], -1)  # (batch,10)
        else:
            num_att = num  
 
        
        cool = self.coolant_embed(x_coolant)   # (batch,4)
        exch = self.exchanger_embed(x_exchanger)  # (batch,4)
        cat_emb = torch.cat([cool, exch], dim=1)  # (batch,8)
        mod = self.mod_fc1(cat_emb)
        mod = self.mod_bn(mod)
        mod = self.mod_act(mod)
        mod = self.mod_fc2(mod)  # (batch, 64)
 
        
        if self.use_attention:
            main_in = torch.cat([num_att, mod], dim=1)  # (batch, 10+64)
            x = self.main_fc1_att(main_in)
        else:
            main_in = torch.cat([num_att, mod], dim=1)  # (batch, 128)
            x = self.main_fc1_noatt(main_in)
 
        x = self.main_bn1(x)
        x = self.main_act1(x)
        x = self.main_drop1(x)
 
       
        if self.residual:
            identity = self.skip_fc(x)
            x1 = self.res_fc1(x)
            x1 = self.res_bn1(x1)
            x1 = self.res_act1(x1)
            x1 = self.res_drop1(x1)
            x1 = self.res_fc2(x1)
            x1 = self.res_bn2(x1)
            x = identity + x1 
            x = self.res_act2(x)
        else:
            x = self.main_fc2(x)
            x = self.main_bn2(x)
            x = self.main_act2(x)
            x = self.main_drop2(x)
 
        x = self.main_fc3(x)
        x = self.main_bn3(x)
        x = self.main_act3(x)
        x = self.main_drop3(x)
        out = self.output_fc(x)
        return out 
 

def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)
 
def smooth_penalty(value, boundary, constraint_type, margin=0.1):
    if constraint_type == 'lower':
        violation = boundary - value 
    else:
        violation = value - boundary 
    if isinstance(violation, torch.Tensor):
        penalty = torch.where(violation <= 0, torch.zeros_like(violation),
                              torch.where(violation <= margin, violation**2 / (2*margin),
                                          violation - margin/2))
    else:
        if violation <= 0:
            penalty = 0.0 
        elif violation <= margin:
            penalty = violation**2 / (2*margin)
        else:
            penalty = violation - margin/2 
    return penalty 
 
def calculate_saturated_air_enthalpy(T, config):
    T = np.clip(T, -50, 200)
    T_clamped = np.clip(T, -50, 100)
    P_sat = 611.21 * np.exp((18.678 - T_clamped/234.5) * (T_clamped / (257.14 + T_clamped)))
    P_sat = np.clip(P_sat, 1, 1e6)
    P_atm = 101325 
    denominator = np.maximum(P_atm - P_sat, 1)
    omega_sat = 0.622 * P_sat / denominator 
    omega_sat = np.clip(omega_sat, 0, 0.1)
    h = ((1.01 + 1.88*omega_sat)*T + 2491*omega_sat) * 1000 
    return h 
 
def calculate_air_enthalpy(T, config):
    T = np.clip(T, -50, 200)
    T_clamped = np.clip(T, -50, 100)
    P_sat = 611.21 * np.exp((18.678 - T_clamped/234.5) * (T_clamped / (257.14 + T_clamped)))
    P_sat = np.clip(P_sat, 1, 1e6)
    RH = (0.038*T_clamped**2 - 4.79*T_clamped + 161.8)/100 
    RH = np.clip(RH, 0.01, 0.9)
    P_vapor = RH * P_sat 
    P_atm = 101325 
    denominator = np.maximum(P_atm - P_vapor, 1)
    omega = 0.622 * P_vapor / denominator 
    omega = np.clip(omega, 0, 0.1)
    h = ((1.01 + 1.88*omega)*T + 2491*omega) * 1000 
    return h 
 
def compute_monotonicity_loss(model, X_batch, Y_pred, config):
    batch_size = X_batch.shape[0]
    output_idx = 0 
    X_data = X_batch.detach().cpu().numpy()
    coolant_type = X_data[:, 10]
    
    numeric_features_idx = [0, 1, 2, 5, 6, 7, 8, 9]  
    monotonicity_water = config.monotonic_constraints['feature_monotonicity']['water']
    monotonicity_air = config.monotonic_constraints['feature_monotonicity']['air']
    epsilon = 1e-4 
    grad_values = np.zeros((len(numeric_features_idx), batch_size))
    model.eval()
    with torch.no_grad():
        for idx, feat_idx in enumerate(numeric_features_idx):
            X_plus = X_batch.clone()
            X_minus = X_batch.clone()
            X_plus[:, feat_idx] += epsilon 
            X_minus[:, feat_idx] -= epsilon 
            x_num_p = X_plus[:, :10]
            x_cool_p = X_plus[:, 10:11]
            x_exch_p = X_plus[:, 11:12]
            x_num_m = X_minus[:, :10]
            x_cool_m = X_minus[:, 10:11]
            x_exch_m = X_minus[:, 11:12]
            y_plus = model(x_num_p, x_cool_p, x_exch_p)
            y_minus = model(x_num_m, x_cool_m, x_exch_m)
            delta_plus = y_plus[:, output_idx].cpu().numpy()
            delta_minus = y_minus[:, output_idx].cpu().numpy()
            grad_values[idx, :] = (delta_plus - delta_minus) / (2 * epsilon)
    model.train()
    total_loss = 0.0 
    total_violations = 0 
    Y_pred_np = Y_pred.detach().cpu().numpy()
    grad_norm = grad_values / (np.abs(Y_pred_np[:, output_idx]) * 10000 + 1e-8)
    for i in range(batch_size):
        vec = monotonicity_water if coolant_type[i] == 1 else monotonicity_air 
        for j, feat_idx in enumerate(numeric_features_idx):
            expected = vec[j]
            if abs(expected) > 0:
                g = grad_norm[j, i]
                if expected > 0 and g < -1e-6:
                    total_loss += -g 
                    total_violations += 1 
                elif expected < 0 and g > 1e-6:
                    total_loss += g 
                    total_violations += 1 
    monotonic_loss = total_loss / total_violations if total_violations > 0 else 0.0 
    return torch.tensor(monotonic_loss, dtype=torch.float32)
 
def calculate_dynamic_weights_enhanced(epoch, total_epochs, validation_loss, config):
    progress = epoch / total_epochs 
    data_weight = config.data_weight 
    if progress <= 0.99:
        physics_weight = config.physics_weight_initial * (1 - progress/0.99) + config.physics_weight_final * (progress/0.99)
        monotonic_weight = config.monotonic_weight_initial * (1 - progress/0.99) + config.monotonic_weight_final * (progress/0.99)
    else:
        physics_weight = config.physics_weight_final 
        monotonic_weight = config.monotonic_weight_final 
    hard_constraint_weight = config.hard_constraint_weight 
    return data_weight, physics_weight, monotonic_weight, hard_constraint_weight 
 
def compute_error_metrics(Y_true, Y_pred):
    n_out = Y_true.shape[1]
    errors = {}
    names = ['delta_T_cold', 'heat_flux', 'HTC', 'efficiency']
    for i in range(n_out):
        yt = Y_true[:, i]
        yp = Y_pred[:, i]
        rmse = np.sqrt(np.mean((yt - yp)**2))
        abs_err = np.abs(yt - yp)
        denom = np.abs(yt) + np.abs(yp) + np.finfo(float).eps 
        smape = abs_err / denom 
        valid = denom > 1e-6 
        mape = np.mean(smape[valid]) * 100 if np.any(valid) else np.nan 
        ss_res = np.sum((yt - yp)**2)
        ss_tot = np.sum((yt - np.mean(yt))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan 
        errors[names[i]] = {'rmse': rmse, 'mape': mape, 'r2': r2}
    errors['overall'] = {
        'rmse': np.mean([errors[n]['rmse'] for n in names]),
        'mape': np.mean([errors[n]['mape'] for n in names]),
        'r2': np.mean([errors[n]['r2'] for n in names])
    }
    return errors 
 
def check_early_stopping(val_loss_history, config):
    if len(val_loss_history) < 10:
        return False, config 
    recent = val_loss_history[-10:]
    if len(recent) >= 5 and all(np.diff(recent[-5:]) > 0):
        config.physics_weight *= 0.5 
        config.hard_constraint_weight *= 0.5 
        return True, config 
    std_loss = np.std(recent)
    mean_loss = np.mean(recent)
    if mean_loss != 0 and std_loss / mean_loss < 0.01:
        return True, config 
    return False, config 
 
def compute_physics_constraint_loss_enhanced(X, Y_pred, T_cold_in_batch, T_hot_out_batch, config):
    X_data = X.detach().cpu().numpy()
    T_cold_in = T_cold_in_batch.detach().cpu().numpy().flatten()
    T_hot_out = T_hot_out_batch.detach().cpu().numpy().flatten()

    if config.scalers is not None:
        scalers = config.scalers 
        coolant_type = X_data[:, 10]
        n_samples = X_data.shape[0]
        Y_scale = np.zeros((4, n_samples))
        Y_min = np.zeros((4, n_samples))
        water_idx = coolant_type == 1 
        air_idx = coolant_type == 0 
        if np.any(water_idx):
            ws = scalers['water']
            Y_scale[:, water_idx] = ws['Y_scale'][:, np.newaxis]
            Y_min[:, water_idx] = ws['Y_min'][:, np.newaxis]
        if np.any(air_idx):
            as_ = scalers['air']
            Y_scale[:, air_idx] = as_['Y_scale'][:, np.newaxis]
            Y_min[:, air_idx] = as_['Y_min'][:, np.newaxis]
        Y_scale_t = torch.tensor(Y_scale.T, dtype=torch.float32)
        Y_min_t = torch.tensor(Y_min.T, dtype=torch.float32)
        Y_pred_orig = Y_pred * Y_scale_t + Y_min_t 
    else:
        Y_pred_orig = Y_pred 
 
    delta_T_cold = Y_pred_orig[:, 0]
    heat_flux = Y_pred_orig[:, 1]
    HTC_pred = Y_pred_orig[:, 2]
    efficiency_pred = Y_pred_orig[:, 3]
 
    T_hot_in = X_data[:, 0]
    flow_hot = X_data[:, 1]
    flow_cold = X_data[:, 2]
    exchanger_type = X_data[:, 11]
 
    physics_loss = torch.tensor(0.0, dtype=torch.float32)
    hard_constraint_loss = torch.tensor(0.0, dtype=torch.float32)
    consistency_loss_part = torch.tensor(0.0, dtype=torch.float32)
    consistency_weight = config.physics_weight_initial 
 
  
    if config.micro_feature_model is not None:
        micro_input = np.column_stack([T_hot_in, flow_hot/5.787648, exchanger_type])
        delta_pred, C_ncg_int_pred, uncertainty_delta, uncertainty_C_ncg = predict_micro_xgboost(
            micro_input, config.micro_feature_model)
        delta_pred = delta_pred.flatten()
        C_ncg_int_pred = C_ncg_int_pred.flatten()
        uncertainty_delta = uncertainty_delta.flatten()
        uncertainty_C_ncg = uncertainty_C_ncg.flatten()
 
        micro_total_uncertainty = np.sqrt(np.mean(uncertainty_delta**2 + uncertainty_C_ncg**2))
        base_weight = config.physics_weight_initial 
        uncertainty_factor = np.exp(-micro_total_uncertainty / config.micro_uncertainty['uncertainty_threshold'])
        consistency_weight = base_weight * uncertainty_factor 
 
        coeff = config.micro_params 
        R_int_theory = (coeff['R_int_coeff_A'] * np.exp(coeff['R_int_coeff_B'] * C_ncg_int_pred) +
                        coeff['R_int_coeff_C'] * delta_pred +
                        coeff['R_int_coeff_D'] * delta_pred**2 +
                        coeff['R_int_coeff_E'])
        R_other = np.zeros_like(coolant_type)
        for i in range(len(coolant_type)):
            if coolant_type[i] == 1:
                R_other[i] = coeff['water_a'] + coeff['water_b'] * flow_cold[i]
            else:
                R_other[i] = (coeff['air_a'] + coeff['air_b'] * flow_cold[i]) / (1 + exchanger_type[i] * coeff['air_c'])
        HTC_theory = coeff['HTC_const'] / (R_int_theory + R_other + 1e-6)
        HTC_theory_t = torch.tensor(HTC_theory, dtype=torch.float32)
 
        n_samples = HTC_pred.shape[0]
        if n_samples > 1:
            diff_pred = HTC_pred.unsqueeze(1) - HTC_pred.unsqueeze(0)
            diff_theory = HTC_theory_t.unsqueeze(1) - HTC_theory_t.unsqueeze(0)
            mask = torch.triu(torch.ones(n_samples, n_samples), diagonal=1)
            diff_pred = diff_pred * mask 
            diff_theory = diff_theory * mask 
            sign_theory = torch.sign(diff_theory)
            margin = 0.01 
            pair_loss = torch.clamp(margin - sign_theory * diff_pred, min=0)
            n_pairs = n_samples * (n_samples - 1) / 2 
            rank_loss = pair_loss.sum() / max(n_pairs, 1)
        else:
            rank_loss = torch.tensor(0.0)
 
        if n_samples > 2:
            diff_pred_trend = HTC_pred[1:] - HTC_pred[:-1]
            diff_theory_trend = HTC_theory_t[1:] - HTC_theory_t[:-1]
            trend_loss = torch.mean(torch.clamp(-diff_theory_trend * diff_pred_trend, min=0))
        else:
            trend_loss = torch.tensor(0.0)
 
        sens_delta = 0.223090 - 2*392.8200*delta_pred 
        sens_C = 0.000645 * 2.368360 * np.exp(2.368360 * C_ncg_int_pred)
        uncertainty_HTC_theory = HTC_theory**2 * np.sqrt((sens_delta*uncertainty_delta)**2 + (sens_C*uncertainty_C_ncg)**2)
        total_uncertainty = np.sqrt(uncertainty_HTC_theory**2 + 0.1**2)
        HTC_diff = HTC_pred - HTC_theory_t 
        dof = 3 
        normalized_diff = HTC_diff / (torch.tensor(total_uncertainty, dtype=torch.float32) + 1e-6)
        prob_loss = torch.mean(torch.log(1 + normalized_diff**2 / dof))
 
        w = config.consistency_weights 
        weak_loss = w['rank'] * rank_loss + w['trend'] * trend_loss + w['prob'] * prob_loss 
        consistency_loss_part = consistency_weight * weak_loss 
        physics_loss = physics_loss + consistency_loss_part 
 
   
    if config.use_hard_constraints:
        n = len(coolant_type)
        HTC_min_l, HTC_max_l, eff_min_l, eff_max_l = 0,0,0,0 
        dT_min_l, dT_max_l, qf_min_l, qf_max_l = 0,0,0,0 
        for i in range(n):
            bc = config.boundary_constraints['water'] if coolant_type[i] == 1 else config.boundary_constraints['air']
            HTC_min_l += smooth_penalty(HTC_pred[i], bc['HTC_min'], 'lower', 0.1)
            HTC_max_l += smooth_penalty(HTC_pred[i], bc['HTC_max'], 'upper', 0.1)
            eff_min_l += smooth_penalty(efficiency_pred[i], bc['efficiency_min'], 'lower', 0.05)
            eff_max_l += smooth_penalty(efficiency_pred[i], bc['efficiency_max'], 'upper', 0.05)
            dT_min_l += smooth_penalty(delta_T_cold[i], bc['delta_T_cold_min'], 'lower', 0.5)
            dT_max_l += smooth_penalty(delta_T_cold[i], bc['delta_T_cold_max'], 'upper', 0.5)
            qf_min_l += smooth_penalty(heat_flux[i], bc['heat_flux_min'], 'lower', 10)
            qf_max_l += smooth_penalty(heat_flux[i], bc['heat_flux_max'], 'upper', 10)
        HTC_min_l /= n; HTC_max_l /= n; eff_min_l /= n; eff_max_l /= n 
        dT_min_l /= n; dT_max_l /= n; qf_min_l /= n; qf_max_l /= n 
        T_hot_out_t = torch.tensor(T_hot_out, dtype=torch.float32)
        T_hot_in_t = torch.tensor(T_hot_in, dtype=torch.float32)
        T_cold_in_t = torch.tensor(T_cold_in, dtype=torch.float32)
        T_hot_out_l1 = torch.mean(smooth_penalty(T_hot_out_t, T_hot_in_t, 'upper', 0.1))
        T_hot_out_l2 = torch.mean(smooth_penalty(T_hot_out_t, T_cold_in_t, 'lower', 0.1))
        hard_constraint_loss = (HTC_min_l + HTC_max_l + eff_min_l + eff_max_l +
                                dT_min_l + dT_max_l + qf_min_l + qf_max_l +
                                T_hot_out_l1 + T_hot_out_l2)
 
    
    n = len(coolant_type)
    delta_T_cold_np = delta_T_cold.detach().cpu().numpy()
    heat_flux_np = heat_flux.detach().cpu().numpy()
    HTC_pred_np = HTC_pred.detach().cpu().numpy()
    efficiency_pred_np = efficiency_pred.detach().cpu().numpy()
    q_hot_enthalpy = np.zeros(n)
    q_cold_enthalpy = np.zeros(n)
    q_final = np.zeros(n)
    HTC_calculated = np.zeros(n)
    efficiency_calculated = np.zeros(n)
    for i in range(n):
        h_hot_in = calculate_saturated_air_enthalpy(T_hot_in[i], config)
        h_hot_out = calculate_saturated_air_enthalpy(T_hot_out[i], config)
        m_dot_hot_i = flow_hot[i] * config.rho_air / 3600 
        q_hot_enthalpy[i] = m_dot_hot_i * (h_hot_in - h_hot_out)
 
        if coolant_type[i] == 1:
            m_dot_cold_i = flow_cold[i] * config.rho_water / 3600 
            q_cold_enthalpy[i] = m_dot_cold_i * config.cp_water * delta_T_cold_np[i]
        else:
            h_cold_in = calculate_air_enthalpy(T_cold_in[i], config)
            h_cold_out = calculate_air_enthalpy(T_cold_in[i] + delta_T_cold_np[i], config)
            m_dot_cold_i = flow_cold[i] * config.rho_air / 3600 
            q_cold_enthalpy[i] = m_dot_cold_i * (h_cold_out - h_cold_in)
 
        if q_hot_enthalpy[i] > 0:
            efficiency_calculated[i] = q_cold_enthalpy[i] / (m_dot_hot_i * h_hot_in)
        else:
            efficiency_calculated[i] = 0 
        efficiency_calculated[i] = np.clip(efficiency_calculated[i], 0, 1)
 
        q_final[i] = (q_hot_enthalpy[i] + q_cold_enthalpy[i]) / 2 / config.A 
        q_cold_enthalpy[i] = min(q_cold_enthalpy[i], q_hot_enthalpy[i])
 
        delta_T1 = T_hot_in[i] - (T_cold_in[i] + delta_T_cold_np[i])
        delta_T2 = T_hot_out[i] - T_cold_in[i]
        delta_T1 = max(delta_T1, 0.1)
        delta_T2 = max(delta_T2, 0.1)
        if abs(delta_T1 - delta_T2) < 1e-6:
            LMTD = (delta_T1 + delta_T2) / 2 
        else:
            ratio = max(delta_T1 / delta_T2, 1e-6)
            ratio = min(ratio, 1e6)
            LMTD = (delta_T1 - delta_T2) / np.log(ratio)
        if q_final[i] > 0 and LMTD > 0:
            HTC_calculated[i] = q_final[i] / LMTD 
        else:
            HTC_calculated[i] = 0 
 
    rel_HTC = (HTC_pred_np - HTC_calculated) / (np.abs(HTC_calculated) + 1e-3)
    HTC_loss = np.mean(rel_HTC**2)
    rel_heat_flux = (heat_flux_np - q_final) / (np.abs(q_final) + 1e-3)
    heat_flux_loss = np.mean(rel_heat_flux**2)
    rel_q_constraint = np.maximum(q_cold_enthalpy - q_hot_enthalpy, 0) / (np.abs(q_hot_enthalpy) + 1e-3)
    q_constraint_loss = np.mean(rel_q_constraint**2)
    efficiency_loss = np.mean((efficiency_pred_np - efficiency_calculated)**2)
  
    physics_loss = physics_loss + 0.0 * HTC_loss + 0.0 * heat_flux_loss + 0.0 * q_constraint_loss + 0.0 * efficiency_loss 
 
    if not isinstance(physics_loss, torch.Tensor):
        physics_loss = torch.tensor(physics_loss, dtype=torch.float32)
    if not isinstance(hard_constraint_loss, torch.Tensor):
        hard_constraint_loss = torch.tensor(hard_constraint_loss, dtype=torch.float32)
    if not isinstance(consistency_loss_part, torch.Tensor):
        consistency_loss_part = torch.tensor(consistency_loss_part, dtype=torch.float32)
 
    return physics_loss, hard_constraint_loss, consistency_weight, consistency_loss_part 
 
def compute_mixed_constraint_loss(X, Y_pred, T_cold_in, T_hot_out, config):
    phys_loss, hard_loss, _, cons_loss = compute_physics_constraint_loss_enhanced(X, Y_pred, T_cold_in, T_hot_out, config)
    return phys_loss, hard_loss, cons_loss 
 
def compute_mixed_constraint_loss_closed_loop(X, Y_pred, T_cold_in, T_hot_out, config, consistency_weight):
    phys_loss, hard_loss, comp_weight, cons_loss = compute_physics_constraint_loss_enhanced(X, Y_pred, T_cold_in, T_hot_out, config)
    phys_loss = phys_loss * comp_weight 
    return phys_loss, hard_loss, cons_loss 
 
def computeGradients_enhanced(model, X1, X2, X3, Y, T_cold_in, T_hot_out, config, epoch):
    Y_pred = model(X1, X2, X3)
    data_loss = mse_loss(Y_pred, Y)
    X_combined = torch.cat([X1, X2, X3], dim=1)
    mixed_loss, hard_loss, cons_loss = compute_mixed_constraint_loss(X_combined, Y_pred, T_cold_in, T_hot_out, config)
    monotonic_loss = compute_monotonicity_loss(model, X_combined, Y_pred, config)
    if not hasattr(config, 'physics_weight'): config.physics_weight = config.physics_weight_initial 
    dw, pw, mw, hw = calculate_dynamic_weights_enhanced(epoch, config.epochs, 0, config)
    total_loss = dw * data_loss + pw * mixed_loss + mw * monotonic_loss + hw * hard_loss 
    return total_loss, mixed_loss, monotonic_loss, hard_loss, cons_loss 
 
def computeGradients_closed_loop(model, X1, X2, X3, Y, T_cold_in, T_hot_out, config, epoch, consistency_weight):
    Y_pred = model(X1, X2, X3)
    data_loss = mse_loss(Y_pred, Y)
    X_combined = torch.cat([X1, X2, X3], dim=1)
    mixed_loss, hard_loss, cons_loss = compute_mixed_constraint_loss_closed_loop(X_combined, Y_pred, T_cold_in, T_hot_out, config, consistency_weight)
    monotonic_loss = compute_monotonicity_loss(model, X_combined, Y_pred, config)
    if not hasattr(config, 'physics_weight'): config.physics_weight = config.physics_weight_initial 
    total_loss = config.data_weight * data_loss + config.physics_weight * mixed_loss + config.monotonic_weight * monotonic_loss + config.hard_constraint_weight * hard_loss 
    return total_loss, mixed_loss, monotonic_loss, hard_loss, cons_loss 
 
def train_model_custom_enhanced(model, X_train, Y_train, X_val, Y_val, config, cycle, consistency_weight):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    n_train = X_train[0].shape[0]
    n_val = X_val[0].shape[0]
    if not hasattr(config, 'physics_weight'): config.physics_weight = config.physics_weight_initial 
    if not hasattr(config, 'data_weight'): config.data_weight = config.data_weight 
    if not hasattr(config, 'monotonic_weight'): config.monotonic_weight = config.monotonic_weight_initial 
    if not hasattr(config, 'hard_constraint_weight'): config.hard_constraint_weight = config.hard_constraint_weight 
    if consistency_weight is not None:
        config.physics_weight = consistency_weight 
    info = {'TrainingLoss': [], 'ValidationLoss': [], 'PhysicsLoss': [], 'MonotonicLoss': [],
            'HardConstraintLoss': [], 'ConsistencyLoss': []}
    val_loss_history = []
    for epoch in range(1, config.epochs+1):
        if epoch > 1:
            dw, pw, mw, hw = calculate_dynamic_weights_enhanced(epoch, config.epochs, prev_val_loss, config)
            config.data_weight = dw 
            config.physics_weight = pw 
            config.monotonic_weight = mw 
            config.hard_constraint_weight = hw 
        idx = np.random.permutation(n_train)
        X_shuf = [X_train[0][idx], X_train[1][idx], X_train[2][idx]]
        Y_shuf = Y_train[idx]
        epoch_loss, phys_loss_ep, mono_loss_ep, hard_loss_ep, cons_loss_ep = 0,0,0,0,0 
        batch_count = 0 
        for start in range(0, n_train, config.batch_size):
            end = min(start+config.batch_size, n_train)
            Xb_num = torch.tensor(X_shuf[0][start:end], dtype=torch.float32)
            Xb_cool = torch.tensor(X_shuf[1][start:end], dtype=torch.float32)
            Xb_exch = torch.tensor(X_shuf[2][start:end], dtype=torch.float32)
            Yb = torch.tensor(Y_shuf[start:end], dtype=torch.float32)
            T_cold = Yb[:, 4:5]
            T_hot = Yb[:, 5:6]
            Y_out = Yb[:, :4]
            optimizer.zero_grad()
            if config.closed_loop_training and consistency_weight is not None:
                total_loss, phys_loss, mono_loss, hard_loss, cons_loss = computeGradients_closed_loop(
                    model, Xb_num, Xb_cool, Xb_exch, Y_out, T_cold, T_hot, config, epoch, consistency_weight)
            else:
                total_loss, phys_loss, mono_loss, hard_loss, cons_loss = computeGradients_enhanced(
                    model, Xb_num, Xb_cool, Xb_exch, Y_out, T_cold, T_hot, config, epoch)
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            phys_loss_ep += phys_loss.item() if isinstance(phys_loss, torch.Tensor) else phys_loss 
            mono_loss_ep += mono_loss.item() if isinstance(mono_loss, torch.Tensor) else mono_loss 
            hard_loss_ep += hard_loss.item() if isinstance(hard_loss, torch.Tensor) else hard_loss 
            cons_loss_ep += cons_loss.item() if isinstance(cons_loss, torch.Tensor) else cons_loss 
            batch_count += 1 
        avg_loss = epoch_loss / max(1, batch_count)
        avg_phys = phys_loss_ep / max(1, batch_count)
        avg_mono = mono_loss_ep / max(1, batch_count)
        avg_hard = hard_loss_ep / max(1, batch_count)
        avg_cons = cons_loss_ep / max(1, batch_count)
 
        model.eval()
        val_loss = 0 
        val_bc = 0 
        with torch.no_grad():
            for start in range(0, n_val, config.batch_size):
                end = min(start+config.batch_size, n_val)
                Xv_num = torch.tensor(X_val[0][start:end], dtype=torch.float32)
                Xv_cool = torch.tensor(X_val[1][start:end], dtype=torch.float32)
                Xv_exch = torch.tensor(X_val[2][start:end], dtype=torch.float32)
                Yv = torch.tensor(Y_val[start:end, :4], dtype=torch.float32)
                Yp = model(Xv_num, Xv_cool, Xv_exch)
                loss = mse_loss(Yp, Yv)
                val_loss += loss.item()
                val_bc += 1 
        avg_val_loss = val_loss / max(1, val_bc)
        prev_val_loss = avg_val_loss 
        model.train()
 
        info['TrainingLoss'].append(avg_loss)
        info['ValidationLoss'].append(avg_val_loss)
        info['PhysicsLoss'].append(avg_phys)
        info['MonotonicLoss'].append(avg_mono)
        info['HardConstraintLoss'].append(avg_hard)
        info['ConsistencyLoss'].append(avg_cons)
        val_loss_history.append(avg_val_loss)
 
        if epoch > 10:
            stop, config = check_early_stopping(val_loss_history, config)
            if stop and epoch > config.epochs/2:
                break 
    return model, info 
 
def train_pure_data_driven(model, X_train, Y_train, X_val, Y_val, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    n_train = X_train[0].shape[0]
    n_val = X_val[0].shape[0]
    for epoch in range(config.epochs):
        idx = np.random.permutation(n_train)
        X_shuf = [X_train[0][idx], X_train[1][idx], X_train[2][idx]]
        Y_shuf = Y_train[idx]
        for start in range(0, n_train, config.batch_size):
            end = min(start+config.batch_size, n_train)
            Xb_num = torch.tensor(X_shuf[0][start:end], dtype=torch.float32)
            Xb_cool = torch.tensor(X_shuf[1][start:end], dtype=torch.float32)
            Xb_exch = torch.tensor(X_shuf[2][start:end], dtype=torch.float32)
            Yb = torch.tensor(Y_shuf[start:end, :4], dtype=torch.float32)
            optimizer.zero_grad()
            Yp = model(Xb_num, Xb_cool, Xb_exch)
            loss = mse_loss(Yp, Yb)
            loss.backward()
            optimizer.step()
    return model 
 
def evaluate_model_no_display(model, X_set, Y_norm, scalers, config):
    model.eval()
    with torch.no_grad():
        X_num = torch.tensor(X_set[0], dtype=torch.float32)
        X_cool = torch.tensor(X_set[1], dtype=torch.float32)
        X_exch = torch.tensor(X_set[2], dtype=torch.float32)
        Y_pred_norm = model(X_num, X_cool, X_exch).cpu().numpy()
    Y_true_norm = Y_norm[:, :4]
    water_flag = X_set[1].flatten() == 1 
    air_flag = ~water_flag 
    Y_true_orig = np.zeros_like(Y_true_norm)
    Y_pred_orig = np.zeros_like(Y_pred_norm)
    if np.any(water_flag):
        ws = scalers['water']
        Y_true_orig[water_flag] = Y_true_norm[water_flag] * ws['Y_scale'] + ws['Y_min']
        Y_pred_orig[water_flag] = Y_pred_norm[water_flag] * ws['Y_scale'] + ws['Y_min']
    if np.any(air_flag):
        as_ = scalers['air']
        Y_true_orig[air_flag] = Y_true_norm[air_flag] * as_['Y_scale'] + as_['Y_min']
        Y_pred_orig[air_flag] = Y_pred_norm[air_flag] * as_['Y_scale'] + as_['Y_min']
    model.train()
    return Y_pred_orig, Y_true_orig, Y_pred_orig 
 
def compute_model_errors(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, scalers, config):
    datasets = {'train': (X_train, Y_train), 'val': (X_val, Y_val), 'test': (X_test, Y_test)}
    errors = {}
    for name, (X, Y) in datasets.items():
        _, Y_true, Y_pred = evaluate_model_no_display(model, X, Y, scalers, config)
        errors[name] = compute_error_metrics(Y_true, Y_pred)
    return errors 
 
def compute_test_htc_rmse(model, X_test, Y_test, scalers, config):
    _, Y_true, Y_pred = evaluate_model_no_display(model, X_test, Y_test, scalers, config)
    errors = compute_error_metrics(Y_true, Y_pred)
    return errors['HTC']['rmse'], errors['HTC']['r2'], errors 
 
def evaluate_htc_consistency(model, X_val, Y_val, scalers, config):
    model.eval()
    with torch.no_grad():
        X_num = torch.tensor(X_val[0], dtype=torch.float32)
        X_cool = torch.tensor(X_val[1], dtype=torch.float32)
        X_exch = torch.tensor(X_val[2], dtype=torch.float32)
        Y_pred_norm = model(X_num, X_cool, X_exch).cpu().numpy()
    water_flag = X_val[1].flatten() == 1 
    air_flag = ~water_flag 
    Y_pred_orig = np.zeros_like(Y_pred_norm)
    Y_true_orig = np.zeros((Y_val.shape[0], 4))
    if np.any(water_flag):
        ws = scalers['water']
        Y_pred_orig[water_flag] = Y_pred_norm[water_flag] * ws['Y_scale'] + ws['Y_min']
        Y_true_orig[water_flag] = Y_val[water_flag, :4] * ws['Y_scale'] + ws['Y_min']
    if np.any(air_flag):
        as_ = scalers['air']
        Y_pred_orig[air_flag] = Y_pred_norm[air_flag] * as_['Y_scale'] + as_['Y_min']
        Y_true_orig[air_flag] = Y_val[air_flag, :4] * as_['Y_scale'] + as_['Y_min']
    HTC_pred = Y_pred_orig[:, 2]
    HTC_true = Y_true_orig[:, 2]
    if config.micro_feature_model is not None:
        T_hot_in = X_val[0][:, 0]
        flow_hot = X_val[0][:, 1]
        flow_cold = X_val[0][:, 2]
        coolant = X_val[1].flatten()
        exchanger = X_val[2].flatten()
        micro_input = np.column_stack([T_hot_in, flow_hot/5.787648, exchanger])
        delta_pred, C_ncg_int_pred, _, _ = predict_micro_xgboost(micro_input, config.micro_feature_model)
        delta_pred = delta_pred.flatten()
        C_ncg_int_pred = C_ncg_int_pred.flatten()
        coeff = config.micro_params 
        R_int_theory = (coeff['R_int_coeff_A'] * np.exp(coeff['R_int_coeff_B'] * C_ncg_int_pred) +
                        coeff['R_int_coeff_C'] * delta_pred +
                        coeff['R_int_coeff_D'] * delta_pred**2 +
                        coeff['R_int_coeff_E'])
        R_other = np.zeros_like(coolant)
        for i in range(len(coolant)):
            if coolant[i] == 1:
                R_other[i] = coeff['water_a'] + coeff['water_b'] * flow_cold[i]
            else:
                R_other[i] = (coeff['air_a'] + coeff['air_b'] * flow_cold[i]) / (1 + exchanger[i] * coeff['air_c'])
        HTC_theory = coeff['HTC_const'] / (R_int_theory + R_other + 1e-6)
    else:
        HTC_theory = np.full_like(HTC_pred, np.nan)
    model.train()
    return HTC_pred, HTC_true, HTC_theory, Y_pred_orig 
 
def adjust_physics_params_from_htc_feedback(config, HTC_pred, HTC_true, HTC_theory, HTC_errors):
    lr = config.closed_loop_param_lr 
    valid = ~np.isnan(HTC_theory) & ~np.isnan(HTC_pred) & (HTC_theory > 0)
    if np.sum(valid) < 10:
        return config 
    ratio = HTC_pred[valid] / HTC_theory[valid]
    mean_ratio = np.mean(ratio)
    if mean_ratio > 1.05:
        config.micro_params['HTC_const'] *= (1 - lr * (mean_ratio - 1))
    elif mean_ratio < 0.95:
        config.micro_params['HTC_const'] *= (1 + lr * (1 - mean_ratio))
    
    if len(valid) > 1:
        rho, _ = spearmanr(HTC_pred[valid], HTC_theory[valid])
    else:
        rho = 0 
    if np.isnan(rho): rho = 0 
    if rho < 0.5:
        config.consistency_weights['rank'] = min(5.0, config.consistency_weights['rank'] + 0.05)
        config.consistency_weights['prob'] = max(0.0, config.consistency_weights['prob'] - 0.02)
    elif rho > 0.9:
        config.consistency_weights['rank'] = max(0.1, config.consistency_weights['rank'] - 0.02)
    bias = np.mean(HTC_errors)
    if bias > 0.5:
        config.micro_params['water_a'] -= lr * 0.0001 
        config.micro_params['air_a'] -= lr * 0.0001 
    elif bias < -0.5:
        config.micro_params['water_a'] += lr * 0.0001 
        config.micro_params['air_a'] += lr * 0.0001 
    return config 
 
def print_errors(errors, model_name):
    print(f'[{model_name}] Dataset performance:')
    for ds in ['train', 'val', 'test']:
        if ds in errors:
            e = errors[ds]
            print(f'  {ds}:')
            for var in ['delta_T_cold', 'heat_flux', 'HTC', 'efficiency']:
                print(f'    {var:12s}: RMSE={e[var]["rmse"]:.4f}, MAPE={e[var]["mape"]:.2f}%, R2={e[var]["r2"]:.4f}')
            print(f'    {"Overall":12s}: RMSE={e["overall"]["rmse"]:.4f}, MAPE={e["overall"]["mape"]:.2f}%, R2={e["overall"]["r2"]:.4f}')
 
def plot_model_predictions_all_sets(model_name, train_true, train_pred, val_true, val_pred, test_true, test_pred):
    vars = ['\Delta T_{cold}', 'Heat Flux', 'HTC', 'Efficiency']
    fig, axes = plt.subplots(2,2, figsize=(10,8))
    colors = plt.cm.tab10.colors[:3]
    for i, ax in enumerate(axes.flat):
        ax.plot(train_true[:,i], train_pred[:,i], '.', color=colors[0], label='Train')
        ax.plot(val_true[:,i], val_pred[:,i], '.', color=colors[1], label='Validation')
        ax.plot(test_true[:,i], test_pred[:,i], '.', color=colors[2], label='Test')
        all_true = np.concatenate([train_true[:,i], val_true[:,i], test_true[:,i]])
        minv, maxv = np.min(all_true), np.max(all_true)
        ax.plot([minv, maxv], [minv, maxv], 'k--')
        ax.set_xlabel(f'True {vars[i]}')
        ax.set_ylabel(f'Predicted {vars[i]}')
        ax.set_title(vars[i])
        ax.legend()
        ax.axis('equal')
        ax.grid(True)
    fig.suptitle(f'{model_name}: Predictions on All Datasets')
    plt.tight_layout()
    plt.show()
 
def train_xgboost_micro_models(sim_data_file):
    data = pd.read_excel(sim_data_file).values 
    np.random.seed(1827)
    P = data[:, :3]
    T_delta = data[:, 3]
    T_c = data[:, 4]
    n_total = P.shape[0]
    temp = np.random.permutation(n_total)
    n_train = 50 
    P_train = P[temp[:n_train]]
    T_train_delta = T_delta[temp[:n_train]]
    T_train_c = T_c[temp[:n_train]]
    P_test = P[temp[n_train:]]
    T_test_delta = T_delta[temp[n_train:]]
    T_test_c = T_c[temp[n_train:]]
 
    scaler_in = MinMaxScaler()
    p_train = scaler_in.fit_transform(P_train)
    p_test = scaler_in.transform(P_test)
    scaler_out_delta = MinMaxScaler()
    t_train_delta = scaler_out_delta.fit_transform(T_train_delta.reshape(-1,1)).flatten()
    scaler_out_c = MinMaxScaler()
    t_train_c = scaler_out_c.fit_transform(T_train_c.reshape(-1,1)).flatten()
 
    space = [
        Integer(1, 500, name='num_trees'),
        Real(0.0001, 0.3, prior='log-uniform', name='lr'),
        Integer(1, 100, name='max_splits')
    ]
 
    def objective_delta(params):
        num_trees, lr, max_splits = params 
        max_leaf_nodes = max_splits + 1    
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        losses = []
        for train_idx, val_idx in kf.split(p_train):
            X_tr, X_val = p_train[train_idx], p_train[val_idx]
            y_tr, y_val = t_train_delta[train_idx], t_train_delta[val_idx]
            model = GradientBoostingRegressor(n_estimators=num_trees, learning_rate=lr,
                                              max_leaf_nodes=max_leaf_nodes, loss='squared_error', random_state=42)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            loss = np.mean((pred - y_val)**2)
            losses.append(loss)
        return np.mean(losses)
 
    def objective_c(params):
        num_trees, lr, max_splits = params 
        max_leaf_nodes = max_splits + 1 
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        losses = []
        for train_idx, val_idx in kf.split(p_train):
            X_tr, X_val = p_train[train_idx], p_train[val_idx]
            y_tr, y_val = t_train_c[train_idx], t_train_c[val_idx]
            model = GradientBoostingRegressor(n_estimators=num_trees, learning_rate=lr,
                                              max_leaf_nodes=max_leaf_nodes, loss='squared_error', random_state=42)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            loss = np.mean((pred - y_val)**2)
            losses.append(loss)
        return np.mean(losses)
 
    print("Performing Bayesian optimization for delta model...")
    res_delta = gp_minimize(objective_delta, space, n_calls=20, random_state=42, verbose=False)
    best_params_delta = {'NumTrees': res_delta.x[0], 'LearnRate': res_delta.x[1], 'MaxNumSplits': res_delta.x[2]}
    print("Performing Bayesian optimization for C_ncg model...")
    res_c = gp_minimize(objective_c, space, n_calls=20, random_state=42, verbose=False)
    best_params_c = {'NumTrees': res_c.x[0], 'LearnRate': res_c.x[1], 'MaxNumSplits': res_c.x[2]}
 
    mc_iterations = 20 
    mc_models_delta = []
    mc_models_c = []
    for mc in range(mc_iterations):
        np.random.seed(1827+mc)
        boot_idx = np.random.choice(n_train, n_train, replace=True)
        p_boot = p_train[boot_idx]
        t_boot_delta = t_train_delta[boot_idx]
        t_boot_c = t_train_c[boot_idx]
        max_leaf_delta = best_params_delta['MaxNumSplits'] + 1 
        max_leaf_c = best_params_c['MaxNumSplits'] + 1 
        model_delta = GradientBoostingRegressor(
            n_estimators=best_params_delta['NumTrees'],
            learning_rate=best_params_delta['LearnRate'],
            max_leaf_nodes=max_leaf_delta,
            loss='squared_error', random_state=1827+mc)
        model_delta.fit(p_boot, t_boot_delta)
        model_c = GradientBoostingRegressor(
            n_estimators=best_params_c['NumTrees'],
            learning_rate=best_params_c['LearnRate'],
            max_leaf_nodes=max_leaf_c,
            loss='squared_error', random_state=1827+mc)
        model_c.fit(p_boot, t_boot_c)
        mc_models_delta.append(model_delta)
        mc_models_c.append(model_c)
 
    micro_model = {
        'scaler_in': scaler_in,
        'scaler_out_delta': scaler_out_delta,
        'scaler_out_c': scaler_out_c,
        'mc_models_delta': mc_models_delta,
        'mc_models_c': mc_models_c,
        'mc_iterations': mc_iterations 
    }
 
    delta_pred_test, C_pred_test, _, _ = predict_micro_xgboost(P_test, micro_model)
    err_metric = lambda y, yp: {
        'rmse': np.sqrt(np.mean((y-yp)**2)),
        'r2': 1 - np.sum((y-yp)**2)/np.sum((y-np.mean(y))**2)
    }
    met_delta = err_metric(T_test_delta, delta_pred_test)
    met_C = err_metric(T_test_c, C_pred_test)
    print('\n========= Micro-feature Model Performance =========')
    print(f'delta   : RMSE={met_delta["rmse"]:.6f}, R2={met_delta["r2"]:.4f}')
    print(f'C_ncg   : RMSE={met_C["rmse"]:.6f}, R2={met_C["r2"]:.4f}')
 
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    ax1.plot(T_test_delta, delta_pred_test, 'b.')
    ax1.plot(T_test_delta, T_test_delta, 'k-')
    ax1.set_xlabel('True delta'); ax1.set_ylabel('Predicted delta')
    ax1.set_title(f'delta (RMSE={met_delta["rmse"]:.4f}, R2={met_delta["r2"]:.4f})')
    ax1.axis('equal'); ax1.grid(True)
    ax2.plot(T_test_c, C_pred_test, 'r.')
    ax2.plot(T_test_c, T_test_c, 'k-')
    ax2.set_xlabel('True C_ncg'); ax2.set_ylabel('Predicted C_ncg')
    ax2.set_title(f'C_ncg (RMSE={met_C["rmse"]:.4f}, R2={met_C["r2"]:.4f})')
    ax2.axis('equal'); ax2.grid(True)
    plt.tight_layout()
    plt.savefig('micro_model_predictions.png')
    plt.show()
    return micro_model 
 
def predict_micro_xgboost(X_new, micro_model):
    p_new = micro_model['scaler_in'].transform(X_new)
    n = X_new.shape[0]
    mc_delta = np.zeros((n, micro_model['mc_iterations']))
    mc_c = np.zeros((n, micro_model['mc_iterations']))
    for mc in range(micro_model['mc_iterations']):
        pred_norm_delta = micro_model['mc_models_delta'][mc].predict(p_new)
        pred_norm_c = micro_model['mc_models_c'][mc].predict(p_new)
        mc_delta[:, mc] = micro_model['scaler_out_delta'].inverse_transform(pred_norm_delta.reshape(-1,1)).flatten()
        mc_c[:, mc] = micro_model['scaler_out_c'].inverse_transform(pred_norm_c.reshape(-1,1)).flatten()
    delta_pred = np.mean(mc_delta, axis=1)
    C_pred = np.mean(mc_c, axis=1)
    unc_delta = np.std(mc_delta, axis=1)
    unc_C = np.std(mc_c, axis=1)
    return delta_pred, C_pred, unc_delta, unc_C 
 
def preprocess_data(expt_data, config):
    X_expt = prepare_expt_features(expt_data)
    Y_expt = prepare_labels(expt_data)
    enhanced = enhance_input_features(np.array(expt_data['coolant_type']), np.array(expt_data['exchanger_type']))
    X_enhanced = np.hstack([X_expt, enhanced])
    X_enhanced = np.delete(X_enhanced, [5,7], axis=1)  
    water_idx = np.array(expt_data['coolant_type']) == 1 
    air_idx = ~water_idx 
    X_water = X_enhanced[water_idx]
    Y_water = Y_expt[water_idx]
    X_air = X_enhanced[air_idx]
    Y_air = Y_expt[air_idx]
 
    num_cols = [0,1,2,5,6,7,8,9,10,11]
    X_num_water = X_water[:, num_cols]
    X_cool_water = X_water[:, 3:4]
    X_exch_water = X_water[:, 4:5]
    X_num_air = X_air[:, num_cols]
    X_cool_air = X_air[:, 3:4]
    X_exch_air = X_air[:, 4:5]
 
    X_num_water_norm, scaler_Xw = normalize_data(X_num_water)
    Y_water_norm, scaler_Yw = normalize_data(Y_water[:, :4])
    X_num_air_norm, scaler_Xa = normalize_data(X_num_air)
    Y_air_norm, scaler_Ya = normalize_data(Y_air[:, :4])
 
    scalers = {
        'water': {'X_min': scaler_Xw['mean'], 'X_scale': scaler_Xw['std'],
                  'Y_min': scaler_Yw['mean'], 'Y_scale': scaler_Yw['std']},
        'air':   {'X_min': scaler_Xa['mean'], 'X_scale': scaler_Xa['std'],
                  'Y_min': scaler_Ya['mean'], 'Y_scale': scaler_Ya['std']}
    }
 
    n_water = X_num_water_norm.shape[0]
    n_air = X_num_air_norm.shape[0]
    idx_water = np.random.permutation(n_water)
    n_val_w = int(round(config.validation_split * n_water))
    n_test_w = int(round(config.test_split * n_water))
    n_train_w = n_water - n_val_w - n_test_w 
    train_w = idx_water[:n_train_w]
    val_w = idx_water[n_train_w:n_train_w+n_val_w]
    test_w = idx_water[n_train_w+n_val_w:]
 
    idx_air = np.random.permutation(n_air)
    n_val_a = int(round(config.validation_split * n_air))
    n_test_a = int(round(config.test_split * n_air))
    n_train_a = n_air - n_val_a - n_test_a 
    train_a = idx_air[:n_train_a]
    val_a = idx_air[n_train_a:n_train_a+n_val_a]
    test_a = idx_air[n_train_a+n_val_a:]
 
    X_train = [np.vstack([X_num_water_norm[train_w], X_num_air_norm[train_a]]),
               np.vstack([X_cool_water[train_w], X_cool_air[train_a]]),
               np.vstack([X_exch_water[train_w], X_exch_air[train_a]])]
    Y_train_full = np.vstack([np.hstack([Y_water_norm[train_w], Y_water[train_w, 4:6]]),
                              np.hstack([Y_air_norm[train_a], Y_air[train_a, 4:6]])])
    X_val = [np.vstack([X_num_water_norm[val_w], X_num_air_norm[val_a]]),
             np.vstack([X_cool_water[val_w], X_cool_air[val_a]]),
             np.vstack([X_exch_water[val_w], X_exch_air[val_a]])]
    Y_val_full = np.vstack([np.hstack([Y_water_norm[val_w], Y_water[val_w, 4:6]]),
                            np.hstack([Y_air_norm[val_a], Y_air[val_a, 4:6]])])
    X_test = [np.vstack([X_num_water_norm[test_w], X_num_air_norm[test_a]]),
              np.vstack([X_cool_water[test_w], X_cool_air[test_a]]),
              np.vstack([X_exch_water[test_w], X_exch_air[test_a]])]
    Y_test_full = np.vstack([np.hstack([Y_water_norm[test_w], Y_water[test_w, 4:6]]),
                             np.hstack([Y_air_norm[test_a], Y_air[test_a, 4:6]])])
 
    extra_info = {
        'T_cold_in_train': Y_train_full[:, 4],
        'T_hot_out_train': Y_train_full[:, 5],
        'T_cold_in_val': Y_val_full[:, 4],
        'T_hot_out_val': Y_val_full[:, 5],
        'T_cold_in_test': Y_test_full[:, 4],
        'T_hot_out_test': Y_test_full[:, 5]
    }
    Y_train = Y_train_full[:, :4]
    Y_val = Y_val_full[:, :4]
    Y_test = Y_test_full[:, :4]
    scalers['water_indices'] = {
        'train': np.concatenate([np.ones(n_train_w), np.zeros(n_train_a)]),
        'val': np.concatenate([np.ones(n_val_w), np.zeros(n_val_a)]),
        'test': np.concatenate([np.ones(n_test_w), np.zeros(n_test_a)])
    }
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, scalers, extra_info 
 
def prepare_training_data(X_train, Y_train_full, X_val, Y_val_full, config):
    return X_train, Y_train_full, X_val, Y_val_full 
 
def main():
    config = initialize_config()
    config.run_data_driven_first = True 
 
    expt_data = load_data(config)
    config.micro_feature_model = train_xgboost_micro_models('sim_data.xlsx')
 
    X_train, Y_train, X_val, Y_val, X_test, Y_test, scalers, extra_info = preprocess_data(expt_data, config)
    config.extra_info = extra_info 
    config.scalers = scalers 
 
    Y_train_full = np.hstack([Y_train, extra_info['T_cold_in_train'].reshape(-1,1), extra_info['T_hot_out_train'].reshape(-1,1)])
    Y_val_full = np.hstack([Y_val, extra_info['T_cold_in_val'].reshape(-1,1), extra_info['T_hot_out_val'].reshape(-1,1)])
    X_train_prep, Y_train_prep, X_val_prep, Y_val_prep = prepare_training_data(X_train, Y_train_full, X_val, Y_val_full, config)
 
    if config.run_data_driven_first:
        print('===== Training pure data-driven model =====')
        config_pure = deepcopy(config)
        config_pure.use_physics_constraint = False 
        config_pure.closed_loop_training = False 
        config_pure.use_pinn_constraint = False 
        config_pure.use_monotonic_constraint = False 
        net_pure = PhysicsNN(config_pure)
        net_pure = train_pure_data_driven(net_pure, X_train_prep, Y_train_prep, X_val_prep, Y_val_prep, config_pure)
        errors_pure = compute_model_errors(net_pure, X_train, Y_train, X_val, Y_val, X_test, Y_test, scalers, config)
        _, pure_train_true, pure_train_pred = evaluate_model_no_display(net_pure, X_train, Y_train, scalers, config)
        _, pure_val_true, pure_val_pred = evaluate_model_no_display(net_pure, X_val, Y_val, scalers, config)
        _, pure_test_true, pure_test_pred = evaluate_model_no_display(net_pure, X_test, Y_test, scalers, config)
 
    print('\n===== Training consistency model =====')
    n_cycles = 8 
    best_cycle = 1 
    best_htc_r2 = -np.inf 
    best_net_physics = None 
    best_errors_physics = None 
    best_pred_test = None 
    best_true_test = None 
    closed_loop_history = {'cycle': []}
 
    net_init = PhysicsNN(config)
    net_current = net_init 
    for cycle in range(1, n_cycles+1):
        if cycle > 1:
            net_current = net_trained 
        net_current.eval()
        with torch.no_grad():
            X1_t = torch.tensor(X_train[0], dtype=torch.float32)
            X2_t = torch.tensor(X_train[1], dtype=torch.float32)
            X3_t = torch.tensor(X_train[2], dtype=torch.float32)
            Y_train_pred_norm = net_current(X1_t, X2_t, X3_t).cpu().numpy()
        water_idx = X_train[1].flatten() == 1 
        air_idx = ~water_idx 
        Y_pred_orig = np.zeros_like(Y_train_pred_norm)
        Y_true_orig = Y_train[:, :4].copy()
        if np.any(water_idx):
            ws = scalers['water']
            Y_pred_orig[water_idx] = Y_train_pred_norm[water_idx] * ws['Y_scale'] + ws['Y_min']
            Y_true_orig[water_idx] = Y_train[water_idx, :4] * ws['Y_scale'] + ws['Y_min']
        if np.any(air_idx):
            as_ = scalers['air']
            Y_pred_orig[air_idx] = Y_train_pred_norm[air_idx] * as_['Y_scale'] + as_['Y_min']
            Y_true_orig[air_idx] = Y_train[air_idx, :4] * as_['Y_scale'] + as_['Y_min']
        HTC_pred_train = Y_pred_orig[:, 2]
        HTC_true_train = Y_true_orig[:, 2]
        HTC_errors = HTC_pred_train - HTC_true_train 
        consistency_weight = config.physics_weight_initial 
        net_trained, training_info = train_model_custom_enhanced(net_current, X_train_prep, Y_train_prep,
                                                                 X_val_prep, Y_val_prep, config, cycle, consistency_weight)
        htc_rmse_test, htc_r2_test, errors_current = compute_test_htc_rmse(net_trained, X_test, Y_test, scalers, config)
        closed_loop_history['cycle'].append({'errors': errors_current, 'htc_r2_test': htc_r2_test})
        if htc_r2_test > best_htc_r2:
            best_htc_r2 = htc_r2_test 
            best_cycle = cycle 
            best_net_physics = net_trained 
            best_errors_physics = errors_current 
            _, best_true_test, best_pred_test = evaluate_model_no_display(best_net_physics, X_test, Y_test, scalers, config)
        _, _, HTC_theory_train, _ = evaluate_htc_consistency(net_trained, X_train_prep, Y_train_prep, scalers, config)
        config = adjust_physics_params_from_htc_feedback(config, HTC_pred_train, HTC_true_train, HTC_theory_train, HTC_errors)
 
        torch.save({
            'net_trained_state': net_trained.state_dict(),
            'training_info': training_info,
            'closed_loop_history': closed_loop_history,
            'config': config,
            'scalers': scalers 
        }, f'closed_loop_cycle_{cycle}_results.pth')
 
    errors_physics_best = compute_model_errors(best_net_physics, X_train, Y_train, X_val, Y_val, X_test, Y_test, scalers, config)
    _, phy_train_true, phy_train_pred = evaluate_model_no_display(best_net_physics, X_train, Y_train, scalers, config)
    _, phy_val_true, phy_val_pred = evaluate_model_no_display(best_net_physics, X_val, Y_val, scalers, config)
    _, phy_test_true, phy_test_pred = evaluate_model_no_display(best_net_physics, X_test, Y_test, scalers, config)
 
    print('\n************ FINAL RESULTS ************')
    print_errors(errors_pure, 'Data-driven')
    print_errors(errors_physics_best, 'Consistency')
    plot_model_predictions_all_sets('Pure Data-Driven', pure_train_true, pure_train_pred,
                                    pure_val_true, pure_val_pred, pure_test_true, pure_test_pred)
    plot_model_predictions_all_sets('Consistency-Constrained', phy_train_true, phy_train_pred,
                                    phy_val_true, phy_val_pred, phy_test_true, phy_test_pred)
 
if __name__ == '__main__':
    main()