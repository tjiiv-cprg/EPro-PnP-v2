from ..builder import DEPTH_CODERS


@DEPTH_CODERS.register_module()
class CamAwareLinearCoder(object):

    def __init__(self,
                 depth_mean=15.0,
                 depth_std=15.0,
                 focal_ref=1260.0):
        super().__init__()
        self.depth_mean = depth_mean
        self.depth_std = depth_std
        self.focal_ref = focal_ref

    def decode(self, pred, focal):
        pred = (pred * self.depth_std + self.depth_mean) * (focal / self.focal_ref)
        return pred
