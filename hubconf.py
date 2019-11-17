dependencies = ['torch','pytorch_transformers']
import torch
import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertConfig,BertForMaskedLM,BertModel,DistilBertTokenizer, DistilBertModel,DistilBertForSequenceClassification

def model3(*args, **kwargs):
    model =MyModel3()
    checkpoint = 'https://sharaf-bucket.s3.amazonaws.com/model-3.dat'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint,map_location=torch.device('cpu'), progress=True))
    return model

def model2(*args, **kwargs):
    model =MyModel2()
    checkpoint = 'https://sharaf-bucket.s3.amazonaws.com/model-2.dat'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint,map_location=torch.device('cpu'), progress=True))
    return model


############################### Model 3  ############################################
class MyModel3(nn.Module):
    def __init__(self, freeze_bert=True):
        super().__init__()
        self.model_version = 3
        # self.static_bert_lyr = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=False)
        self.bert_lyr = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

        self.a_attn = nn.Linear(768, 1)

        self.c_attn = nn.Linear(768, 1)

        self.attn_dropout = nn.Dropout(0.1)

        self.ctx_transfomer = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, 768), nn.LayerNorm(768))

        self.a_switch = nn.Sequential(nn.Linear(768 * 2, 1), nn.Sigmoid())
        self.c_switch = nn.Sequential(
            nn.Linear(768 * 2, 1),
            nn.Sigmoid(),
        )

        self.action_cls_lyr = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 64, bias=True),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 4, bias=True),
            nn.LayerNorm(4),
            # nn.Linear(768,len(action_le.classes_),bias=False),
        )

        self.component_cls_lyr = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 64, bias=True),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 5, bias=True),
            nn.LayerNorm(5),
            # nn.Linear(768,len(component_le.classes_),bias=False),
        )

        # for p in self.static_bert_lyr.parameters():
        #   p.requires_grad = False

        # Freeze bert layers
        if freeze_bert:
            for lyr in self.bert_lyr.encoder.layer[:-2]:
                for p in lyr.parameters():  # self.bert_lyr.parameters():
                    p.requires_grad = False
        # nn.init.xavier_uniform_(self.action_cls_lyr)
        # nn.init.xavier_uniform_(self.component_cls_lyr)

    def forward(self, seq, attn_masks, output_attn=False, output_hs=False):
        attn_mask_cls = (1 - attn_masks) * -10000
        attn_mask_cls.unsqueeze_(dim=-1)

        # static_emb,static_ctx = self.static_bert_lyr(seq,attention_mask =attn_masks)
        seq_emb, ctx, hs = self.bert_lyr(seq, attention_mask=attn_masks)
        ctx = self.ctx_transfomer(ctx.detach())
        # seq_emb +=static_emb
        a = self.a_attn(seq_emb)
        a = a + attn_mask_cls
        a = a_output = a.softmax(dim=1)
        a = self.attn_dropout(a)
        a = torch.mul(seq_emb, a)
        a = a.mean(dim=1)

        c = self.c_attn(seq_emb)
        c = c + attn_mask_cls
        c = c_output = c.softmax(dim=1)
        c = self.attn_dropout(c)
        c = torch.mul(seq_emb, c)
        c = c.mean(dim=1)

        a_switch = self.a_switch(torch.cat([ctx, a], dim=-1))
        a = a_switch * a + (1 - a_switch) * ctx

        c_switch = self.c_switch(torch.cat([ctx, c], dim=-1))
        c = c_switch * c + (1 - c_switch) * ctx

        outputs = [self.action_cls_lyr(a), self.component_cls_lyr(c)]
        if (output_attn):
            outputs += [a_output, c_output]
        if output_hs:
            outputs += [hs]
        return outputs


############################### Model 2  ############################################

class MyModel2(nn.Module):
    def __init__(self, freeze_bert=True):
        super().__init__()
        self.model_version = 2 - 1
        # self.static_bert_lyr = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=False)
        self.bert_lyr = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

        self.a_attn = nn.Linear(768, 1)

        self.c_attn = nn.Linear(768, 1)

        self.action_cls_lyr = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 64, bias=True),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, 4, bias=True),
            nn.LayerNorm(4),
            # nn.Linear(768,len(action_le.classes_),bias=False),
        )

        self.component_cls_lyr = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 64, bias=True),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, 5, bias=True),
            nn.LayerNorm(5),
            # nn.Linear(768,len(component_le.classes_),bias=False),
        )

        # for p in self.static_bert_lyr.parameters():
        #   p.requires_grad = False

        # Freeze bert layers
        if freeze_bert:
            for lyr in self.bert_lyr.encoder.layer[:-2]:
                for p in lyr.parameters():  # self.bert_lyr.parameters():
                    p.requires_grad = False
        # nn.init.xavier_uniform_(self.action_cls_lyr)
        # nn.init.xavier_uniform_(self.component_cls_lyr)

    def forward(self, seq, attn_masks, output_attn=False, output_hs=False):
        attn_mask_cls = (1 - attn_masks) * -10000
        attn_mask_cls.unsqueeze_(dim=-1)

        # static_emb,static_ctx = self.static_bert_lyr(seq,attention_mask =attn_masks)
        seq_emb, ctx, hs = self.bert_lyr(seq, attention_mask=attn_masks)
        # seq_emb +=static_emb
        a = self.a_attn(seq_emb)
        a = a + attn_mask_cls
        a = a_output = a.softmax(dim=1)
        a = torch.mul(seq_emb, a)
        a = a.mean(dim=1)

        c = self.c_attn(seq_emb)
        c = c + attn_mask_cls
        c = c_output = c.softmax(dim=1)
        c = torch.mul(seq_emb, c)
        c = c.mean(dim=1)

        outputs = [self.action_cls_lyr(a), self.component_cls_lyr(c)]
        if (output_attn):
            outputs += [a_output, c_output]
        if output_hs:
            outputs += [hs]
        return outputs
#####################################################################################