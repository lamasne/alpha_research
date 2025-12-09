Parameters of the NN used for the heatmap "grid_search_patience_lr_map":
- epochs=100,
- train_frac=0.8, 
- is_standardize=False, 
- is_lr_scheduler=True, 
lr is learning rate and patience is early stoping patience.

And NN parameters
hidden_dim = 64, 
dropout = 0.3:
self.net = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.BatchNorm1d(hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),

    nn.Linear(hidden_dim, hidden_dim),
    nn.BatchNorm1d(hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),

    nn.Linear(hidden_dim, 1),
)