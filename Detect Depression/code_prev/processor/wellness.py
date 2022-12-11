import os
import copy
import json
import torch
from torch.utils.data import TensorDataset

class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """
    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    
class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class WellnessRegProcessor(object):
    """Processor for the Wellness data set """
    def __init__(self, args):
        self.args = args

    @classmethod
    def _read_file(cls, input_file):
        """
        Reads a tab separated value file (csv).
        f: data/wellness/Wellness_Conversation_intent_train.tsv 
        """
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                tmp = line.strip(',') 
                lines.append(tmp[:-1]) 
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[1:]):
            line = line.split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]   # reg: 0, cls: 1
            label = line[1]   # reg: 1, 2, 3, cls: 2
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        return self._create_examples(
            self._read_file(os.path.join(self.args.data_dir, self.args.task, file_to_read)), mode
        )


class WellnessClsProcessor(object):
    """Processor for the Wellness data set """
    def __init__(self, args):
        self.args = args

    def get_labels(self):
        label_list = list(range(19))
        label_list = list(map(str, label_list)) 
        return label_list 

    @classmethod
    def _read_file(cls, input_file):
        """
        Reads a tab separated value file (csv).
        f: data/wellness/Wellness_Conversation_intent_train.tsv 
        """
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                tmp = line.strip(',') 
                lines.append(tmp[:-1]) 
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[1:]):
            line = line.split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = line[2]   # reg: 0, cls: 2
            label = line[3]   # reg: 1, 2, 3, cls: 3
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        return self._create_examples(
            self._read_file(os.path.join(self.args.data_dir, self.args.task, file_to_read)), mode
        )

seq_reg_processors = {
    "wellness": WellnessRegProcessor,
}

seq_cls_processors = {
    "wellness": WellnessClsProcessor,
}

seq_cls_tasks_num_labels = {"wellness": 19}

seq_reg_output_modes = {
    "wellness": "regression", 
}

seq_cls_output_modes = {
    "wellness": "classification",
}

def seq_reg_convert_examples_to_features(args, examples, tokenizer, max_length, task):
    processor = seq_reg_processors[task](args)
    def label_from_example(example):   # regression
        return float(example.label)
        
    labels = [label_from_example(example) for example in examples]
    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        add_special_tokens=True,
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = [0] * len(inputs["input_ids"])  # For xlm-roberta
            
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)
    return features

def seq_cls_convert_examples_to_features(args, examples, tokenizer, max_length, task):
    processor = seq_cls_processors[task](args)
    label_list = processor.get_labels()
    output_mode = seq_cls_output_modes[task]
    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example):
        return label_map[example.label]

    labels = [label_from_example(example) for example in examples]
    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        add_special_tokens=True,
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = [0] * len(inputs["input_ids"])  # For xlm-roberta

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)
    return features

def seq_reg_load_and_cache_examples(args, tokenizer, mode):
    processor = seq_reg_processors[args.task](args)  

    # Load data features from cache or dataset file
    if mode == "train":
        examples = processor.get_examples("train")
    elif mode == "dev":
        examples = processor.get_examples("dev")
    elif mode == "test":
        examples = processor.get_examples("test")
    else:
        raise ValueError("For mode, only train, dev, test is avaiable")

    features = seq_reg_convert_examples_to_features(
            args, examples, tokenizer, max_length=args['max_seq_len'], task='wellness'
      )
    
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def seq_cls_load_and_cache_examples(args, tokenizer, mode):
    processor = seq_cls_processors[args.task](args)   # WellnessProcessor(args)
    output_mode = seq_cls_output_modes[args.task]   # classification 
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,   # data 
        "cached_{}_{}_{}_{}".format(
            str(args.task), list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_len), mode
        ),
    )
    
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise ValueError("For mode, only train, dev, test is avaiable")
        features = seq_cls_convert_examples_to_features(
            args, examples, tokenizer, max_length=args.max_seq_len, task=args.task
        )
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset