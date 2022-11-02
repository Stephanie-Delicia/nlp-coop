"""
Stephanie M.
Teacher forced generations
"""
import logging
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForSequenceClassification
import numpy as np
import torch
from transformers.modeling_utils import PreTrainedModel
from typing import List, Optional

logging.basicConfig(format='%(asctime)s %(message)s')

"""
A class representing a model which is an encapsulation of a generation and a sentiment model.
During generation, the model chooses the top n generation logits and computes their
sentiment scores with the sentiment model. Logits/words are filtered by sentiment
scores that are at least within epsilon distance from the expected sentiment score.  
"""
class DualModel(PreTrainedModel):
  def __init__(self, 
  config, 
  generation_model, 
  auxillary_model, 
  epsilon: float, 
  num_candidates: int, 
  pos_label_index = None, 
  estimated_score = None):
    """
    generation_model - (str) generation/base model (must have a standard HF forward, e.g. from BartForConditionalGeneration)
    auxillary_model  - (str) sentiment model for filtering low sentiment score words 
    epsilon          - threshold in [0, 1] that filter logits based on the distance between sentiment score and true label
    num_candidates   - number of top logits to select from the generation model's logits output
    pos_label_index  - index of positive label from classification logits base on label2id mapping
    estimated_score  - estimated sentiment score of generation
    """
    super().__init__(config)
    self.generation_model = AutoModelForSeq2SeqLM.from_pretrained(generation_model)
    self.auxillary_model = AutoModelForSequenceClassification.from_pretrained(auxillary_model)
    self.epsilon = epsilon
    self.num_candidates = num_candidates
    self.lm_head = self.generation_model.lm_head
    self.config = config
    self.pos_label_index = pos_label_index
    self.estimated_score = estimated_score

  def get_encoder(self):
      return self.generation_model.get_encoder()

  def get_decoder(self):
      return self.generation_model.get_decoder()
    
  def forward(self,
        input_ids: torch.LongTensor = None,                       # input prompts to be summarized         
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,     
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None, 
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        current_seq_ids_per_beam: Optional[torch.LongTensor] = None, # Tokens ID's of output seq. produced so far
        return_dict: Optional[bool] = True,):
    base_outputs = self.generation_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,)
    n = self.num_candidates
    print("Generation model: " + str(self.generation_model.device))
    print("auxillary model: " + str(self.auxillary_model.device))
    # returns array of indices of n top logit values per beam
    new_logits = base_outputs.logits.clone().cpu()
    top_logit_indices = []
    for i in range(new_logits.shape[0]): 
        top_logits = np.argpartition(new_logits[i][0], -n)[-n:]
        top_logits = top_logits.to(self.device)
        top_logit_indices.append(top_logits.tolist()) 
    new_logits = new_logits.to(self.device)
    #logging.warning("Obtained top n words per beam.")

    # Create list of the possible sequences with every top w_i candidate (w_i + w_{i - 1}...) per beam
    # TODO: I think this part is running very slowly..
    sentiment_scores_seq_candidates = [] 
    pos_index = self.pos_label_index
    for i in range(len(current_seq_ids_per_beam)):
        candidate_seqs = []
        for cand in top_logit_indices[i]:
            cand_seq = torch.cat((current_seq_ids_per_beam[i], torch.tensor([cand]).to(self.device)), 0) # create candidate seq.
            cand_seq = torch.tensor([cand_seq.tolist()]).to(self.device)
            #logging.warning("Created a candidate sequence.")
            cand_seq_score = torch.softmax(self.auxillary_model(cand_seq).logits, dim=1).to(self.device).tolist()[0][pos_index]
            #logging.warning("Scored the sequence.")
            candidate_seqs.append(cand_seq_score) # append score of this seq. of this beam
        sentiment_scores_seq_candidates.append(candidate_seqs)
    #logging.warning("Created and scored sentiment of candidate sequences.")

    # Filter logits by sentiment scores
    final_candidate_indices_per_beam = [] # list of candidate indices per beam
    for beam in range(len(sentiment_scores_seq_candidates)):
        top_indices_of_this_beam = top_logit_indices[beam]                               # indices of top logits per beam
        beam_scores = sentiment_scores_seq_candidates[beam]                              # sentiment score per seq. of beam
        filtered_indices = [top_indices_of_this_beam[i] for i in range(len(beam_scores)) # filter by |score - estimated_score| <= epsilon
        if abs(beam_scores[i] - self.estimated_score) <= self.epsilon]
        final_candidate_indices_per_beam.append(filtered_indices)
    #logging.warning("Computed sentiment scores and obtained final candidate words.")

    # Set filtered logits to -10000
    mask = -10000
    for beam in range(len(new_logits)):
        for index in range(len(new_logits[beam][0])):               # access logits per beam
            if index not in final_candidate_indices_per_beam[beam]: # if index of logit is not in candidate indices
                new_logits[beam][0][index] = mask
    logging.warning("Finished filtering round.")

    # torch.save(new_logits,  'new_logits')
    # torch.save(base_outputs,  'base_outputs')
    # torch.save(base_outputs.logits,  'base-logits')
    # torch.save(decoder_input_ids,  'decoder_input_ids')
    # torch.save(top_logit_indices,  'top_logit_indices')
    # torch.save(current_seq_ids_per_beam,  'current_seq_ids_per_beam')
    # torch.save(sentiment_scores_seq_candidates,  'sentiment_scores_seq_candidates')
    # torch.save(final_candidate_indices_per_beam,  'final_candidate_indices_per_beam')
    base_outputs.logits = new_logits
    return base_outputs

  # stolen from BART! 
  def prepare_inputs_for_generation(
    self,
    decoder_input_ids,
    past=None,
    attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
    use_cache=None,
    encoder_outputs=None,
    **kwargs
):
    current_seq_ids_per_beam = decoder_input_ids.clone()
    if past is not None:
        decoder_input_ids = decoder_input_ids[:, -1:]

    return {
        "input_ids": None,  # encoder_outputs is defined. input_ids not needed
        "current_seq_ids_per_beam": current_seq_ids_per_beam,
        "encoder_outputs": encoder_outputs,
        "past_key_values": past,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
        "use_cache": use_cache, 
    }

  @staticmethod  
  def _reorder_cache(past, beam_idx):
    reordered_past = ()
    for layer_past in past:
        # cached cross_attention states don't have to be reordered -> they are always the same
        reordered_past += (
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
        )
    return reordered_past