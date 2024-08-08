class BaseAugmentation:
    def __init__(self, resize, mean, std, **kwargs):
        self.transform = Compose([
        ])

    def __call__(self, image):
        return self.transform(image)