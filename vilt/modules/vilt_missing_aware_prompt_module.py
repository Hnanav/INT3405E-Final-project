import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

class ViLTransformerSS(pl.LightningModule):
    def __init__(self):
        super().__init__()

        bert_config = BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=768 * 4,
            max_position_embeddings=20,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(init_weights)

        self.token_type_embeddings = nn.Embedding(2, 768)
        self.token_type_embeddings.apply(init_weights)


        self.transformer = vit_base_patch16_224(pretrained=True, num_classes = 18).to("cuda")

        self.pooler = Pooler(768)
        self.pooler.apply(init_weights)

        hs = 768


        cls_num = 18
        self.mmimdb_classifier = nn.Sequential(
            nn.Linear(hs, hs * 2),
            nn.LayerNorm(hs * 2),
            nn.GELU(),
            nn.Linear(hs * 2, cls_num),
        )
        self.mmimdb_classifier.apply(init_weights)


        self.prompt_type = 'input'
        prompt_length = 16

        self.prompt_length = prompt_length
        embed_dim = 768
        self.learnt_p = True
        self.prompt_layers = [0,1,2,3,4,5]
        self.multi_layer_prompt = True

        prompt_num = len(self.prompt_layers) if self.multi_layer_prompt else 1

        complete_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        complete_prompt[:,0:1,:].fill_(1)
        if self.learnt_p and self.prompt_type == 'attention':
            complete_prompt[:,prompt_length//2:prompt_length//2+1,:].fill_(1)
        self.complete_prompt = nn.Parameter(complete_prompt)

        missing_text_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_text_prompt[:,2:3,:].fill_(1)
        if self.learnt_p and self.prompt_type == 'attention':
            missing_text_prompt[:,prompt_length//2+2:prompt_length//2+3,:].fill_(1)
        self.missing_text_prompt = nn.Parameter(missing_text_prompt)

        missing_img_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_img_prompt[:,1:2,:].fill_(1)
        if self.learnt_p and self.prompt_type == 'attention':
            missing_img_prompt[:,prompt_length//2+1:prompt_length//2+2,:].fill_(1)
        self.missing_img_prompt = nn.Parameter(missing_img_prompt)

        if not self.learnt_p:
            self.complete_prompt.requires_grad=False
            self.missing_text_prompt.requires_grad=False
            self.missing_img_prompt.requires_grad=False

        print(self.complete_prompt)
        print(self.missing_img_prompt)
        print(self.missing_text_prompt)

        for param in self.transformer.parameters():
            param.requires_grad=True
        for param in self.text_embeddings.parameters():
            param.requires_grad=True
        for param in self.token_type_embeddings.parameters():
            param.requires_grad=False

        self.current_tasks = list()


    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        is_train=None,
    ):


        text_ids = batch[f"text_ids"]
        text_labels = batch[f"label"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        img = batch["image"]

        if image_embeds is None and image_masks is None:

            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=15,
                mask_it=mask_image,
            )

        else:
            patch_index, image_labels = (
                None,
                None,
            )

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        # instance wise missing aware prompts
        prompts = None
        for idx in range(len(img)):
            if batch["missing_type"][idx] == 0:
                prompt = self.complete_prompt
            elif batch["missing_type"][idx] == 1:
                prompt = self.missing_text_prompt
            elif batch["missing_type"][idx] == 2:
                prompt = self.missing_img_prompt

            if prompt.size(0) != 1:
                prompt = prompt.unsqueeze(0)

            if prompts is None:
                prompts = prompt
            else:
                prompts = torch.cat([prompts, prompt], dim=0)

        if self.learnt_p:
            if self.prompt_type=='attention':
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length//2, dtype=prompts.dtype, device=prompts.device).long()
            elif self.prompt_type=='input':
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length*len(self.prompt_layers), dtype=prompts.dtype, device=prompts.device).long()
        else:
            prompt_masks = torch.ones(prompts.shape[0], self.prompt_length, dtype=prompts.dtype, device=prompts.device).long()

        co_masks = torch.cat([prompt_masks, text_masks, image_masks], dim=1)
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        x = co_embeds.detach()

        for i, blk in enumerate(self.transformer.blocks):
            if i in self.prompt_layers:
                if self.multi_layer_prompt:
                    x, _attn = blk(x, mask=co_masks,
                                   prompts=prompts[:,self.prompt_layers.index(i)],
                                   learnt_p=self.learnt_p,
                                   prompt_type=self.prompt_type)
                else:
                    x, _attn = blk(x, mask=co_masks, prompts=prompts, learnt_p=self.learnt_p)
            else:
                x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)

        if self.prompt_type == 'input':
            total_prompt_len = len(self.prompt_layers)* prompts.shape[-2]
        elif self.prompt_type == 'attention':
            total_prompt_len = prompts.shape[-2]

        text_feats, image_feats = (
            x[:,total_prompt_len : total_prompt_len+text_embeds.shape[1]],
            x[:, total_prompt_len+text_embeds.shape[1] :],
        )
        if self.prompt_type == 'input':
            cls_feats = self.pooler(x[:,total_prompt_len:total_prompt_len+1])
#         cls_feats = self.pooler(x[:,:prompts.size(1)].mean(dim=1,keepdim=True))
        elif self.prompt_type == 'attention':
            cls_feats = self.pooler(x)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }
        return ret

    def forward(self, batch):
        ret = dict()

        # Multi-label classification for MM-IMDb
        if "mmimdb" in self.current_tasks:
            ret.update(compute_mmimdb(self, batch))

        ret.update(compute_mmimdb(self, batch))
        return ret

    def training_step(self, batch, batch_idx):
        output = self(batch)
        total_loss = output['mmimdb_loss']

        self.log("train_loss", total_loss.item())
        self.log("train_f1", output['f1'])


        return total_loss


    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self(batch)
            total_loss = output['mmimdb_loss']
            self.log("valid_loss", total_loss.item(), on_epoch=True)
            self.log("valid_f1", output['f1'], on_epoch=True)
            return total_loss


    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self(batch)
            total_loss = output['mmimdb_loss']
            self.log("test_loss", total_loss.item(), on_epoch=True)
            self.log("test_f1", output['f1'], on_epoch=True)

        return total_loss
    def train_dataloader(self):
        return train_dataloader
    def valid_dataloader(self):
        return val_dataloader
    def test_dataloader(self):
        return test_dataloader
    def configure_optimizers(self):
        lr = 1e-4
        wd = 0.01

        no_decay = [
            "bias",
            "LayerNorm.bias",
            "LayerNorm.weight",
            "norm.bias",
            "norm.weight",
            "norm1.bias",
            "norm1.weight",
            "norm2.bias",
            "norm2.weight",
        ]
        head_names = ["vqa_classifier", "mmimdb_classifier", "food101_classifier", "hatememes_classifier", "nlvr2_classifier"]
        prompt_name = "prompt"
        lr_mult = 1
        end_lr = 0
        decay_power = 1
        optim_type = "adamw"

        names = [n for n, p in self.named_parameters()]


        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                ],
                "weight_decay": wd,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(bb in n for bb in head_names)
                ],
                "weight_decay": wd,
                "lr": lr * lr_mult,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
                ],
                "weight_decay": 0.0,
                "lr": lr * lr_mult,
            },
        ]

        if optim_type == "adamw":
            optimizer = transformers.optimization.AdamW(
                optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
            )
        elif optim_type == "adam":
            optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
        elif optim_type == "sgd":
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

        if self.trainer.max_steps is None:
            max_steps = (
                len(self.trainer.datamodule.train_dataloader())
                * self.trainer.max_epochs
                // self.trainer.accumulate_grad_batches
            )
        else:
            max_steps = self.trainer.max_steps

        warmup_steps = 2500

        if decay_power == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
            )
        else:
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
                lr_end=end_lr,
                power=decay_power,
            )

        sched = {"scheduler": scheduler, "interval": "step"}

        return (
            [optimizer],
            [sched],
        )

