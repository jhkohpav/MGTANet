from ..registry import DETECTORS
from .single_stage import SingleStageDetector, SingleStageDetectorVID
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy 
import time

@DETECTORS.register_module
class MGTANet(SingleStageDetectorVID):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        alignment=None,
        aggregation=None,
        seq_len = None,
        sequence_mode = None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(MGTANet, self).__init__(
            reader, backbone, neck, alignment, aggregation,
            bbox_head, train_cfg, test_cfg, pretrained
        )
        self.seq_len = seq_len
        self.sequence_mode = sequence_mode
        self.total_time = 0
        self.iters = 0

    def extract_feat(self, data):
        input_features = self.reader(data["features"], data["num_voxels"])
        x, voxel_feature = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x, ms_x = self.neck(x, True)

        return x, ms_x, voxel_feature

    def forward(self, example, return_loss=True, **kwargs):
        voxels_all = example["voxels"]
        coordinates_all = example["coordinates"]
        num_points_in_voxel_all = example["num_points"]
        num_voxels_all = example["num_voxels"]

        batch_size = len(num_voxels_all[0])

        x_all = []
        ms_x_all = []
        for seq in range(self.seq_len):
            data = dict(
                features=voxels_all[seq],
                num_voxels=num_points_in_voxel_all[seq],
                coors=coordinates_all[seq],
                batch_size=batch_size,
                input_shape=example["shape"][0],
            )
            start_time = time.time()
            x, ms_x, _ = self.extract_feat(data)
            x_all.append(x)
            ms_x_all.append(ms_x)
        
        if self.alignment is not None:
            aligned_x_all = self.alignment(x_all, ms_x_all)
        else:
            aligned_x_all = x_all
        
        if self.aggregation is not None:
            aggregated_x = self.aggregation(aligned_x_all)
        else:
            aggregated_x = aligned_x_all[0]
        
        preds = self.bbox_head(aggregated_x)

        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            boxes = self.bbox_head.predict(example, preds, self.test_cfg)
            end_time = time.time()
            self.total_time += (end_time - start_time)
            self.iters += 1
            #print('avg_speed', self.total_time/self.iters)
            return boxes
