from pep import PEPEmbedding

from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.dfm import DeepFactorizationMachineModel
from torchfm.model.afi import AutomaticFeatureInteractionModel


def get_model(name, dataset, embed_dim,
              dfm_mlp_dims=(16, 16), dfm_drop_out=0.2,
              afi_mlp_dims=(400, 400), afi_drop_out=(0, 0, 0)):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    if name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=embed_dim)
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims, embed_dim=embed_dim,
                                             mlp_dims=dfm_mlp_dims, dropout=dfm_drop_out)
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(field_dims, embed_dim=embed_dim,
                                                atten_embed_dim=64, num_heads=2,
                                                num_layers=3, mlp_dims=afi_mlp_dims, dropouts=afi_drop_out)
    else:
        raise ValueError('unknown model name: ' + name)


def trans_pep(model, field_dims, embed_dim,
              retrain=False, emb_save_path=None, retrain_emb_param=0,
              threshold_type='feature_dim', threshold_init=-15,
              g_type='sigmoid', gk=1):
    pep = PEPEmbedding(threshold_type, latent_dim=embed_dim, field_dim=field_dims,
                       retrain=retrain, emb_save_path=emb_save_path, retrain_emb_param=retrain_emb_param,
                       g_type=g_type, gk=gk,
                       threshold_init=threshold_init)
    model.embedding = pep
