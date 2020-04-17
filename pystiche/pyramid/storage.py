from pystiche.ops import Comparison

__all__ = ["ImageStorage"]


class ImageStorage:
    def __init__(self, ops):
        self.target_guides = {}
        self.target_images = {}
        self.input_guides = {}
        for op in ops:
            if isinstance(op.cls, Comparison):
                try:
                    self.target_guides[op] = op.target_guide
                except AttributeError:
                    pass

                try:
                    self.target_images[op] = op.target_image
                except AttributeError:
                    pass

            try:
                self.input_guides[op] = op.input_guide
            except AttributeError:
                pass

    def restore(self):
        # self._clear_encoding_storage()

        for op, target_guide in self.target_guides.items():
            op.set_target_guide(target_guide, recalc_repr=False)

        for op, target_image in self.target_images.items():
            op.set_target_image(target_image)

        for op, input_guide in self.input_guides.items():
            op.set_input_guide(input_guide)

    # def _clear_encoding_storage(self):
    #     ops = set(
    #         itertools.chain(
    #             self.target_guides.keys(),
    #             self.target_images.keys(),
    #             self.input_guides.keys(),
    #         )
    #     )
    #     encoding_ops = [op for op in ops if isinstance(op, EncodingOperator)]
    #     multi_layer_encoders = set(
    #         [
    #             op.encoder.multi_layer_encoder
    #             for op in encoding_ops
    #             if isinstance(op.encoder, SingleLayerEncoder)
    #         ]
    #     )
    #
    #     for multi_layer_encoder in multi_layer_encoders:
    #         multi_layer_encoder.clear_cache()
