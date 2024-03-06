import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer
from torch.nn import TransformerDecoder
# from src.dlutils import *
from scipy import stats
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        if self.batch_first:
            pe = pe.unsqueeze(0)
        else: 
            pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        if self.batch_first:
            x = x + self.pe[pos:pos+x.size(1), :]
        else:
            x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)

class TransformerBasic(nn.Module):
	def __init__(self, feats):
		super().__init__()
		self.name = 'TransformerBasic'
		self.lr = 0.1
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10

		self.lin = nn.Linear(1, feats)
		self.out_lin = nn.Linear(feats, 1)
		self.pos_encoder = PositionalEncoding(feats, 0.1, feats*self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()

	def forward(self, src, tgt):
		# bs x (ws x features) x features
		src = src * np.sqrt(self.n_feats)
		src = self.lin(src.unsqueeze(2))
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)

		tgt = tgt * np.sqrt(self.n_feats)
		tgt = self.lin(tgt.unsqueeze(2))
		tgt = self.pos_encoder(tgt)
		x = self.transformer_decoder(tgt, memory)
		x = self.out_lin(x)
		x = self.fcn(x)
		return x

class TransformerBasicv2(nn.Module):
	def __init__(self, feats, lr, window_size):
		super(TransformerBasicv2, self).__init__()
		self.name = 'TransformerBasicv2'
		self.lr = lr
		# self.batch = 128
		self.batch = 64
		self.n_feats = feats
		self.n_window = window_size
		self.scale = 16
		self.linear_layer = nn.Linear(feats, self.scale*feats)
		self.output_layer = nn.Linear(self.scale*feats, feats)
		self.pos_encoder = PositionalEncoding(self.scale*feats, 0.1,
										 self.n_window, batch_first=True)
		encoder_layers = TransformerEncoderLayer(d_model=feats*self.scale, nhead=feats,
										    batch_first=True, dim_feedforward=256, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats*self.scale, nhead=feats,
										    batch_first=True, dim_feedforward=256, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()

	def forward(self, src, tgt):
		src = src * np.sqrt(self.n_feats)
		src = self.linear_layer(src)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src) 

		tgt = tgt * np.sqrt(self.n_feats)
		tgt = self.linear_layer(tgt)

		x = self.transformer_decoder(tgt, memory)
		x = self.output_layer(x)
		x = self.fcn(x)
		return x

class TransformerBasicv2Scaling(nn.Module):
	def __init__(self, feats, lr, window_size):
		super(TransformerBasicv2Scaling, self).__init__()
		self.name = 'TransformerBasicv2Scaling'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = window_size
		self.scale = 16
		self.linear_layer = nn.Linear(feats, self.scale*feats)
		self.output_layer = nn.Linear(self.scale*feats, feats)
		self.pos_encoder = PositionalEncoding(self.scale*feats, 0.1, self.n_window, batch_first=True)
		encoder_layers = TransformerEncoderLayer(d_model=feats*self.scale, nhead=feats, batch_first=True, dim_feedforward=256, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats*self.scale, nhead=feats, batch_first=True, dim_feedforward=256, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()

	def forward(self, src, tgt):
		model_dim = self.scale * self.n_feats

		src = self.linear_layer(src)
		src = src * np.sqrt(model_dim)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src) 

		tgt = self.linear_layer(tgt)
		tgt = tgt * np.sqrt(model_dim)

		x = self.transformer_decoder(tgt, memory)
		x = self.output_layer(x)
		x = self.fcn(x)
		return x



class TransformerBasicBottleneck(nn.Module):
	def __init__(self, feats, window_size):
		super(TransformerBasicBottleneck, self).__init__()
		self.name = 'TransformerBasicBottleneck'
		# self.lr = lr
		self.batch = 16
		self.n_feats = feats
		self.n_window = window_size
		self.scale = 16
		self.linear_layer = nn.Linear(feats, self.scale*feats)
		self.output_layer = nn.Linear(self.scale*feats, feats)
		self.pos_encoder = PositionalEncoding(self.scale*feats, 0.1, self.n_window, batch_first=True)
		encoder_layers = TransformerEncoderLayer(d_model=feats*self.scale, nhead=feats,
										    batch_first=True, dim_feedforward=256, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats*self.scale, nhead=feats,
										    batch_first=True, dim_feedforward=256, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()

	def forward(self, src, tgt):
		src = src * np.sqrt(self.n_feats)
		src = self.linear_layer(src)
		src = self.pos_encoder(src)
		# batch x t x d
		memory = self.transformer_encoder(src) 
		# batch x 1 x d
		z = torch.mean(memory, dim=1, keepdim=True)


		tgt = tgt * np.sqrt(self.n_feats)
		tgt = self.linear_layer(tgt)

		x = self.transformer_decoder(tgt, z)
		x = self.output_layer(x)
		x = self.fcn(x)
		return x

class TransformerBasicBottleneckScaling(nn.Module):
	def __init__(self, feats, lr, window_size, batch_size):
		super(TransformerBasicBottleneckScaling, self).__init__()
		self.name = 'TransformerBasicBottleneckScaling'
		self.lr = lr
		self.batch = batch_size
		self.n_feats = feats
		self.n_window = window_size
		self.scale = 16
		self.linear_layer = nn.Linear(feats, self.scale*feats)
		self.output_layer = nn.Linear(self.scale*feats, feats)
		self.pos_encoder = PositionalEncoding(self.scale*feats, 0.1, self.n_window, batch_first=True)
		encoder_layers = TransformerEncoderLayer(d_model=feats*self.scale,  # nhead=feats,
										    batch_first=True, dim_feedforward=256, dropout=0.1) 
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats*self.scale, nhead=feats, batch_first=True, dim_feedforward=256, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()

	def forward(self, src, tgt):
		model_dim = self.scale * self.n_feats

		src = self.linear_layer(src)
		src = src * np.sqrt(model_dim)
		src = self.pos_encoder(src)
		# batch x t x d
		memory = self.transformer_encoder(src) 
		# batch x 1 x d
		z = torch.mean(memory, dim=1, keepdim=True)

		tgt = self.linear_layer(tgt)
		tgt = tgt * np.sqrt(model_dim)

		x = self.transformer_decoder(tgt, z)
		x = self.output_layer(x)
		x = self.fcn(x)
		return x
