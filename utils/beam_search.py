import torch.nn.functional as f
import torch
import torch.nn as nn
import operator
from queue import PriorityQueue 
from .util import idxs2sentences
sos_token = 3 
eos_token = 4
max_length = 20

class BeamSearchNode(object): 
    def __init__(self, hidden, previousnode, wordid, logprob, length):

        self.h = hidden
        self.prevnode = previousnode
        self.wordid = wordid
        self.logp = logprob
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

def beam_decode(decoder ,batch_size, decoder_hiddens, beam_width=3):

    topk = 1
    decoded_batch = torch.zeros(batch_size, max_length, dtype=torch.long)    
    lens = []
    # decoding goes sentence by sentence
    for idx in range(batch_size):
        if isinstance(decoder_hiddens, tuple): # lstm case
            decoder_hidden = (decoder_hiddens[0][:,idx,:].unsqueeze(0),decoder_hiddens[1][:, idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        
        # start with the start of the sentence token
        decoder_input = torch.LongTensor([[sos_token]]).to(decoder_hidden[0].device)
        endnodes = []
        number_required = 3 
        # starting node - hidden vector, previous node, work id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 2, 1)
        nodes = PriorityQueue()
        # start the queue
        nodes.put((-node.eval(),1,node))
        qsize = 1
        index = 1
        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize == 4000:
                print(qsize)
            if qsize > 4000 : break
            # fetch the best node
            score, _,n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h
            if n.wordid.item() == eos_token and n.prevnode != None:
                endnodes.append((score, n))
                #if we reached maximum of sentence required
                if len(endnodes) >= number_required:
                    
                    break
                else:
                    continue
            if n.leng >= max_length:
                endnodes.append((score, n))
                if len(endnodes) >= number_required:
                    break
                else:
                    continue
            # decode for one step using decoder
            with torch.no_grad():
                decoder_output, decoder_hidden = decoder.one_step(decoder_input, decoder_hidden[0], decoder_hidden[1])
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []
            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()
                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score,index+1, nn))
                index += 1
                # increase qsize
            qsize += len(nextnodes) - 1
        
        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            for _ in range(topk):
                score,_,n = nodes.get() 
                endnodes.append((score,n))  
        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            if n.wordid != eos_token:
                utterance.append(n.wordid.item())
            # back trace
            while n.prevnode != None:
                n = n.prevnode
                if n.prevnode != None:
                    utterance.append(n.wordid.item())

            utterance = utterance[::-1]
            utterances.append(utterance)
        for i in utterances:
            if len(i)> 0: 
                utter =torch.tensor(i)
                break
    
        decoded_batch[idx,:utter.size(0)] = utter 
        lens.append(utter.size(0))
    return decoded_batch, torch.tensor(lens)
