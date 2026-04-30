import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

def calc_entropy(input_tensor):
    lsm = nn.LogSoftmax(dim=-1)
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum()
    return entropy

class MyT5ForConditionalGenerationScore(T5ForConditionalGeneration):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Gọi luồng chạy gốc chuẩn của Transformers
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs
        )
        
        lm_logits = outputs.logits
        loss = None
        
        # Tính toán lại Loss và Entropy cho riêng bài toán ABSA (giống hệt logic cũ)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
            loss_list = []
            entropy_list = []
            for i in range(lm_logits.size(0)):
                loss_i = loss_fct(lm_logits[i], labels[i])
                if decoder_attention_mask is not None:
                    valid_len = int(decoder_attention_mask[i].sum().item())
                else:
                    valid_len = int((labels[i] != -100).sum().item())
                
                ent = calc_entropy(lm_logits[i, :valid_len])
                loss_list.append(loss_i.item())
                entropy_list.append(ent.item())
            loss = [loss_list, entropy_list]
            outputs.loss = loss
        
        if not return_dict:
            return (loss, lm_logits) + outputs[2:] if loss is not None else (lm_logits,) + outputs[1:]
        
        return outputs
