_base_ = [
    "../_base_/common.py",
    "../_base_/train.py",
    "../_base_/test.py",
]

has_test = True
deterministic = True
use_custom_worker_init = False
model_name = "MFFN"

train = dict(
    batch_size=8,
    num_workers=4,
    use_amp=True,
    num_epochs=50,
    epoch_based=True,
    lr=0.05,
    optimizer=dict(
        mode="sgd",
        set_to_none=True,
        group_mode="finetune",
        cfg=dict(
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=False,
        ),
    ),
    sche_usebatch=True,
    scheduler=dict(
        warmup=dict(
            num_iters=0,
            initial_coef=0.01,
            mode="linear",
        ),
        mode="f3",
        cfg=dict(
            lr_decay=0.9,
            min_coef=0.001,
        ),
    ),
)

test = dict(
    batch_size=8,
    num_workers=4,
    show_bar=False,
)

datasets = dict(
    train=dict(
        dataset_type="MFFN_cod_tr",
        shape=dict(h=384, w=384),
        path=["cod10k_camo_tr"],
        interp_cfg=dict(),
    ),
    test=dict(
        dataset_type="MFFN_cod_te",
        shape=dict(h=384, w=384),
#         path=["cpd1k_te", "cod10k_te", "nc4k"],
        path=["nc4k"],
        interp_cfg=dict(),
    ),
)
