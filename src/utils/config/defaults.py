"""Default configuration values."""

DEFAULT_CONFIG = {
    'model': {
        'latent_dim': 256,
        'action_dim': 4,
        'hidden_dim': 512,
        'num_layers': 4
    },
    'training': {
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'use_amp': False
    },
    'data': {
        'sequence_length': 10,
        'image_size': [256, 256]
    }
}