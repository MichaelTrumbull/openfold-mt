# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
THIS IS A MODIFIED VERSION OF HEADS.PY
SEE HEADS_ORIGINAL.PY
MODIFIED FOR SECOND TO LAST LAYER ACTIVATION INVESTIGATION
'''

## test modification line...
## another test mod


import torch
import torch.nn as nn

from openfold.model.primitives import Linear, LayerNorm
from openfold.utils.loss import (
    compute_plddt,
    compute_tm,
    compute_predicted_aligned_error,
)
'''
def qprint(classname, location, item):
    print('-'*8)
    print(classname, location)
    #print(item)
    if type(item) is dict: print(item.keys())
    if torch.is_tensor(item): print(item.size())
    # save tensor to colab
    torch.save(item, '/content/drive/My Drive/Colab Notebooks/' + classname + location + ".pt")

def #modify(second_to_last_layer):
    print('will #modify: ', second_to_last_layer.size())
    
    try:
        nudge_value = torch.load('/content/drive/My Drive/Colab Notebooks/nudge_value.pt').item()
        print('Using nudge_value: ', nudge_value)
    except:
        print('File not found: /content/drive/My Drive/Colab Notebooks/nudge_value.pt')
        print('Setting nudge_value to 1')
        nudge_value = 1
    
    modified_second_to_last_layer = torch.mul(second_to_last_layer, 1)#nudge_value)
    
    return modified_second_to_last_layer
'''

class AuxiliaryHeads(nn.Module):
    def __init__(self, config):
        super(AuxiliaryHeads, self).__init__()

        self.plddt = PerResidueLDDTCaPredictor(
            **config["lddt"],
        )

        self.distogram = DistogramHead(
            **config["distogram"],
        )

        self.masked_msa = MaskedMSAHead(
            **config["masked_msa"],
        )

        self.experimentally_resolved = ExperimentallyResolvedHead(
            **config["experimentally_resolved"],
        )

        if config.tm.enabled:
            self.tm = TMScoreHead(
                **config.tm,
            )

        self.config = config

    def forward(self, outputs):
        aux_out = {}
        lddt_logits = self.plddt(outputs["sm"]["single"])
        aux_out["lddt_logits"] = lddt_logits

        # Required for relaxation later on
        aux_out["plddt"] = compute_plddt(lddt_logits)

        distogram_logits = self.distogram(outputs["pair"])
        aux_out["distogram_logits"] = distogram_logits

        masked_msa_logits = self.masked_msa(outputs["msa"])
        aux_out["masked_msa_logits"] = masked_msa_logits

        experimentally_resolved_logits = self.experimentally_resolved(
            outputs["single"]
        )
        aux_out[
            "experimentally_resolved_logits"
        ] = experimentally_resolved_logits

        if self.config.tm.enabled:
            tm_logits = self.tm(outputs["pair"])
            aux_out["tm_logits"] = tm_logits
            aux_out["predicted_tm_score"] = compute_tm(
                tm_logits, **self.config.tm
            )
            aux_out.update(
                compute_predicted_aligned_error(
                    tm_logits,
                    **self.config.tm,
                )
            )
        print('heads.py/AuxillaryHeads: aux_out.keys()')
        print(aux_out.keys())

        return aux_out


class PerResidueLDDTCaPredictor(nn.Module):
    def __init__(self, no_bins, c_in, c_hidden):
        super(PerResidueLDDTCaPredictor, self).__init__()

        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden

        self.layer_norm = LayerNorm(self.c_in)

        self.linear_1 = Linear(self.c_in, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_3 = Linear(self.c_hidden, self.no_bins, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)

        ##qprint("PerResidueLDDTCaPredictor", "Before", s)
        # ##modify latent space to search for other conformations

        s = self.linear_3(s)

        ##qprint("PerResidueLDDTCaPredictor", "After", s)

        return s


class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.

    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(self, c_z, no_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(self.c_z, self.no_bins, init="final")

    def forward(self, z):  # [*, N, N, C_z]
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """
        # [*, N, N, no_bins]

        #qprint("DistogramHead", "Before", z)

        logits = self.linear(z)

        #qprint("DistogramHead", "middle", logits)

        #logits = modify(logits) #this might NOT be the right location for this...

        logits = logits + logits.transpose(-2, -3)

        #qprint("DistogramHead", "After", logits)

        return logits


class TMScoreHead(nn.Module):
    """
    For use in computation of TM-score, subsection 1.9.7
    """

    def __init__(self, c_z, no_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of bins
        """
        super(TMScoreHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(self.c_z, self.no_bins, init="final")

    def forward(self, z):
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pairwise embedding
        Returns:
            [*, N_res, N_res, no_bins] prediction
        """
        # [*, N, N, no_bins]

        #qprint("TMScoreHead", "Before", z)

        #z = modify(z)

        logits = self.linear(z)

        #qprint("TMScoreHead", "After", logits)

        return logits


class MaskedMSAHead(nn.Module):
    """
    For use in computation of masked MSA loss, subsection 1.9.9
    """

    def __init__(self, c_m, c_out, **kwargs):
        """
        Args:
            c_m:
                MSA channel dimension
            c_out:
                Output channel dimension
        """
        super(MaskedMSAHead, self).__init__()

        self.c_m = c_m
        self.c_out = c_out

        self.linear = Linear(self.c_m, self.c_out, init="final")

    def forward(self, m):
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
        Returns:
            [*, N_seq, N_res, C_out] reconstruction
        """
        # [*, N_seq, N_res, C_out]

        #qprint("MaskedMSAHead", "Before", m)

        #m = modify(m)

        logits = self.linear(m)

        #qprint("MaskedMSAHead", "After", logits)

        return logits


class ExperimentallyResolvedHead(nn.Module):
    """
    For use in computation of "experimentally resolved" loss, subsection
    1.9.10
    """

    def __init__(self, c_s, c_out, **kwargs):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of distogram bins
        """
        super(ExperimentallyResolvedHead, self).__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, init="final")

    def forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] single embedding
        Returns:
            [*, N, C_out] logits
        """
        # [*, N, C_out]

        #qprint("ExperimentallyResolvedHead", "Before", s)

        #s = modify(s)
        
        logits = self.linear(s)

        #qprint("ExperimentallyResolvedHead", "After", logits)

        return logits
