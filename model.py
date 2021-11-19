
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaPreTrainedModel

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput

@dataclass
class QuestionAnsweringModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class MultiModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    doc_logits : torch.FloatTensor = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class ConvLayer(nn.Module) :
    def __init__(self, seq_size, feature_size, intermediate_size) :
        super(ConvLayer, self).__init__()
        self.conv_layer = nn.Sequential(nn.Conv1d(seq_size, intermediate_size, 5, padding=2),
            nn.Conv1d(intermediate_size, seq_size, 1),
            nn.ReLU()
        )
        self.layer_norm = nn.LayerNorm(feature_size, eps=1e-6)
            
    def forward(self, x) :
        y = x + self.conv_layer(x)
        y = self.layer_norm(y)
        return y

class ConvNetHead(nn.Module) :
    def __init__(self, layer_size, seq_size, feature_size, intermediate_size) :
        super(ConvNetHead, self).__init__()
        conv_net = [ConvLayer(seq_size, feature_size, intermediate_size) for i in range(layer_size)]
        self.conv_net = nn.Sequential(*conv_net)
        self.init_weights()

    def init_weights(self) :
        for p in self.parameters() :
            if p.requires_grad == True and p.dim() > 1:
                nn.init.kaiming_uniform_(p)

    def forward(self, x) :
        y = self.conv_net(x)
        return y

class LSTMHead(nn.Module) :
    def __init__(self, layer_size, feature_size, intermediate_size) :
        super(LSTMHead, self).__init__()
        self.layer_size = layer_size
        self.feature_size = feature_size
        self.intermediate_size = intermediate_size

        self.lstm = torch.nn.LSTM(input_size=feature_size,
            hidden_size = intermediate_size,
            num_layers=layer_size,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        self.init_weights()

    def init_weights(self) :
        for p in self.parameters() :
            if p.requires_grad == True and p.dim() > 1:
                nn.init.kaiming_uniform_(p)

    def forward(self, x) :
        batch_size = x.shape[0]

        h_input = torch.zeros((2*self.layer_size, batch_size, self.intermediate_size))
        c_input = torch.zeros((2*self.layer_size, batch_size, self.intermediate_size))

        if torch.cuda.is_available() :
            h_input = h_input.cuda()
            c_input = c_input.cuda()

        y, (h_output, c_output) = self.lstm(x, (h_input,c_input))
        return y


class SDSNetForQuestionAnswering(RobertaPreTrainedModel):
    def __init__(self, model_name, data_args, config):
        super(SDSNetForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = RobertaModel.from_pretrained(model_name, 
            config=config, 
            add_pooling_layer=False
        )

        self.cnn_head = ConvNetHead(layer_size=data_args.layer_size, 
            seq_size=data_args.max_seq_length,
            feature_size=config.hidden_size,
            intermediate_size=data_args.intermediate_size
        )

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0] # (batch_size, seq_size, hidden_size) : [CLS] Token
        sequence_output = self.cnn_head(sequence_output)

        logits = self.qa_outputs(sequence_output) # (batch_size, seq_size, label_size=2)
        start_logits, end_logits = logits.split(1, dim=-1)  
        start_logits = start_logits.squeeze(-1).contiguous() # (batch_size, seq_size) 
        end_logits = end_logits.squeeze(-1).contiguous() # (batch_size, seq_size)

        total_loss = None
        # start_positions : (batch_size, )
        # end_positions : (batch_size, )
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # make answer token logits bigger, find answer position
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index) 
            start_loss = loss_fct(start_logits, start_positions) 
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LSTMForQuestionAnswering(RobertaPreTrainedModel):
    def __init__(self, model_name, data_args, config):
        super(LSTMForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels
        self.output_size = data_args.intermediate_size * 2
        self.bert = RobertaModel.from_pretrained(model_name, 
            config=config, 
            add_pooling_layer=False
        )

        self.lstm_head = LSTMHead(layer_size=data_args.layer_size,
            feature_size=config.hidden_size,
            intermediate_size=data_args.intermediate_size
        )
        
        self.qa_outputs = nn.Linear(self.output_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0] # (batch_size, seq_size, hidden_size) : [CLS] Token
        sequence_output = self.lstm_head(sequence_output)

        logits = self.qa_outputs(sequence_output) # (batch_size, seq_size, label_size=2)
        start_logits, end_logits = logits.split(1, dim=-1)  
        start_logits = start_logits.squeeze(-1).contiguous() # (batch_size, seq_size) 
        end_logits = end_logits.squeeze(-1).contiguous() # (batch_size, seq_size)

        total_loss = None
        # start_positions : (batch_size, )
        # end_positions : (batch_size, )
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # make answer token logits bigger, find answer position
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index) 
            start_loss = loss_fct(start_logits, start_positions) 
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




class MultiNetForQuestionAnswering(RobertaPreTrainedModel):
    def __init__(self, model_name, data_args, config):
        super(MultiNetForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = RobertaModel.from_pretrained(model_name, 
            config=config, 
            add_pooling_layer=False
        )

        self.flag_outputs = nn.Linear(config.hidden_size, 1)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        doc_flags=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0] # (batch_size, seq_size, hidden_size) : [CLS] Token

        cls_output = sequence_output[:,0] # (batch_size, hidden_size)
        flags = self.flag_outputs(cls_output) # (batch_size, 1)
        flags_logits = torch.sigmoid(flags.squeeze(-1))
        
        flags_loss = None
        if doc_flags is not None :
            loss_bin = nn.BCELoss()
            flags_loss = loss_bin(flags_logits, doc_flags.float())

        logits = self.qa_outputs(sequence_output) # (batch_size, seq_size, label_size=2)
        start_logits, end_logits = logits.split(1, dim=-1)  
        start_logits = start_logits.squeeze(-1).contiguous() # (batch_size, seq_size) 
        end_logits = end_logits.squeeze(-1).contiguous() # (batch_size, seq_size)

        total_loss = None
        # start_positions : (batch_size, )
        # end_positions : (batch_size, )
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # make answer token logits bigger, find answer position
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index) 
            start_loss = loss_fct(start_logits, start_positions) 
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if doc_flags is not None :
                total_loss += flags_loss
    
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MultiModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            doc_logits=flags_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

