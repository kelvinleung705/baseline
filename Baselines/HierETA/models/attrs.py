import torch
import torch.nn as nn


class Attr(nn.Module):
    # Pre-encoded 7D global features from CSV
    ext_cates = []
    ext_conts = ["global_features"]

    # Segment Features (Static JSON + Live CSV)
    seg_cates = [
        ("segID", 1376566 + 1, 16),
        ("segment_functional_level", 10 + 1, 2),
        ("laneNum", 10 + 1, 2),
        ("roadLevel", 10 + 1, 2),
    ]
    seg_conts = ["wid", "speedLimit", "time", "len", "roadState"]

    # NO LINK FEATURES!
    link_cates = []
    link_conts = []

    def __init__(self, FLAGS):
        super(Attr, self).__init__()
        self.batch_size = FLAGS.batch_size

        for name, dim_in, dim_out in (
            Attr.ext_cates + Attr.seg_cates + Attr.link_cates
        ):
            self.add_module("attr-" + name, nn.Embedding(dim_in, dim_out))

    def forward(self, attrs):
        ext = self.emb_helper(attrs, "ext")
        seg = self.emb_helper(attrs, "seg")
        link = self.emb_helper(attrs, "link")
        return ext, seg, link

    def emb_helper(self, attrs, type):
        Cates, Conts = Attr.type_helper(type)
        emb_list = []

        # 1. Categorical Embeddings
        for name, dim_in, dim_out in Cates:
            if name in attrs:
                embed = getattr(self, "attr-" + name)
                attr_t = attrs[name].view(self.batch_size, -1)
                attr_t = embed(attr_t)
                emb_list.append(attr_t)

        # 2. Continuous Features
        for name in Conts:
            if name in attrs:
                attr_t = attrs[name].float()
                if type == "seg":
                    if attr_t.dim() == 2:
                        attr_t = attr_t.unsqueeze(-1)
                    elif attr_t.dim() == 3:
                        attr_t = attr_t.view(self.batch_size, -1).unsqueeze(-1)
                emb_list.append(attr_t)

        # SAFEGUARD FOR LINK: If no link features exist, return dummy zero tensor (Batch, 3, 1)
        if not emb_list:
            link_num = attrs["road_link_mask"].size(1)  # 3 links
            device = attrs["ext"].device
            return torch.zeros((self.batch_size, link_num, 1), device=device)

        out = torch.cat(emb_list, -1)
        return out

    @staticmethod
    def out_size(type):
        Cates, Conts = Attr.type_helper(type)
        size = 0
        for name, dim_in, dim_out in Cates:
            size += dim_out

        if type == "ext":
            size += 7  # 7 global features
        elif type == "seg":
            size += len(Conts)
        elif type == "link":
            size = 1  # 1 dummy dimension for link

        return size

    @staticmethod
    def type_helper(types):
        if types == "ext":
            Cates = Attr.ext_cates
            Conts = Attr.ext_conts
        elif types == "seg":
            Cates = Attr.seg_cates
            Conts = Attr.seg_conts
        elif types == "link":
            Cates = Attr.link_cates
            Conts = Attr.link_conts
        else:
            raise Exception("must choose from ext, seg and link!")
        return Cates, Conts