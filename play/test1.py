from allennlp.modules.elmo import Elmo, batch_to_ids
import ipdb

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway" \
               "/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway" \
              "/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)

# use batch_to_ids to convert sentences to character ids
#sentences = [['First', 'sentence', 'is', 'me', '.'], ['Another', '.']]
sentences = [['First', 'sentence', 'is', 'me', '.'], ['Ha']]

character_ids = batch_to_ids(sentences)

# embeddings['elmo_representations'][1].detach().numpy().shape -> numpy

embeddings = elmo(character_ids)

# ipdb> embeddings['elmo_representations'][1].shape
# torch.Size([1, 5, 1024])

ipdb.set_trace()