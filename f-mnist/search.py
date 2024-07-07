import os
import copy
import json
import numpy as np
from stylegan.renderer import Renderer
from config import STYLEGAN_INIT, SEARCH_LIMIT, STYLEMIX_SEED_LIMIT, INIT_PKL, STYLEMIX_LAYERS, FRONTIER_PAIRS
from predictor import Predictor
from utils import validate_mutation


class mimicry:
    def __init__(self, class_idx=None, w0_seed=0, stylemix_seed=0, search_limit=SEARCH_LIMIT , step_size=1):
        self.state = STYLEGAN_INIT
        self.class_idx = class_idx
        self.w0_seed = w0_seed
        self.stylemix_seed = stylemix_seed
        self.search_limit = search_limit
        self.stylemix_seed_limit = STYLEMIX_SEED_LIMIT
        self.layers = STYLEMIX_LAYERS
        self.step_size = step_size
        self.state['renderer'] = Renderer()
        # self.res = dnnlib.EasyDict()

    def render_state(self, state=None):
        if state is None:
            state = self.state
        state['renderer']._render_impl(
            res = state['generator_params'],  # res
            pkl = INIT_PKL,  # pkl
            w0_seeds= state['params']['w0_seeds'],  # w0_seed,
            class_idx = state['params']['class_idx'],  # class_idx,
            mixclass_idx = state['params']['mixclass_idx'],  # mix_idx,
            stylemix_idx = state['params']['stylemix_idx'],  # stylemix_idx,
            stylemix_seed = state['params']['stylemix_seed'],  # stylemix_seed,
            img_normalize = state['params']['img_normalize'],
            to_pil = state['params']['to_pil'],
        )

        info =  copy.deepcopy(state['params'])

        return state, info

    def search(self):
        root = f"{FRONTIER_PAIRS}/{self.class_idx}/"

        frontier_seed_count = 0
        while frontier_seed_count < self.search_limit:
            state = self.state

            state["params"]["class_idx"] = self.class_idx
            state["params"]["w0_seeds"] = [[self.w0_seed, 1.0]]
            state["params"]["stylemix_idx"] = []
            state["params"]["mixclass_idx"] = None
            state["params"]["stylemix_seed"] = None

            digit, digit_info = self.render_state()


            label = digit["params"]["class_idx"]
            image = digit['generator_params'].image
            image = image.crop((2, 2, image.width - 2, image.height - 2))
            image_array = np.array(image)

            accepted, confidence, predictions = Predictor().predict_datapoint(
                np.reshape(image_array, (-1, 28, 28, 1)),
                label
            )
            # print(f"Accepted: {accepted} - Confidence: {confidence} - Predictions: {predictions}")

            digit_info["accepted"] = accepted.tolist()
            digit_info["exp-confidence"] = float(confidence)
            digit_info["predictions"] = predictions.tolist()

            if accepted:
                found_at_least_one = False
                _ , second_cls = np.argsort(-predictions)[:2]
                second_cls_confidence = predictions[second_cls]
                if second_cls_confidence:
                    for stylemix_cls, cls_confidence in enumerate(predictions):
                        if stylemix_cls != label and cls_confidence:
                            # found mutation below threshold
                            found_mutation = False
                            tried_all_layers = False

                            state["params"]["mixclass_idx"] = stylemix_cls
                            self.stylemix_seed = 0

                            while not found_mutation and not tried_all_layers and self.stylemix_seed < self.stylemix_seed_limit:

                                # require unique seed for each stylemix
                                if self.stylemix_seed == self.w0_seed:
                                    self.stylemix_seed += 1
                                state["params"]["stylemix_seed"] = self.stylemix_seed

                                for idx, layer in enumerate(self.layers):
                                    state["params"]["stylemix_idx"] = layer

                                    m_digit, m_digit_info = self.render_state()
                                    m_image = m_digit['generator_params'].image
                                    m_image = m_image.crop((2, 2, m_image.width - 2, m_image.height - 2))
                                    m_image_array = np.array(m_image)


                                    m_accepted, confidence , m_predictions = Predictor().predict_datapoint(
                                        np.reshape(m_image_array, (-1, 28, 28, 1)),
                                        label
                                    )

                                    m_class = np.argsort(-m_predictions)[:1]
                                    # misclassification and decision boundary check
                                    if not m_accepted and stylemix_cls == m_class:

                                        valid_mutation, ssi, l2_distance, img_l2, m_img_l2 = validate_mutation(image_array, m_image_array)

                                        if valid_mutation:
                                            if not found_at_least_one:
                                                frontier_seed_count += 1
                                                found_at_least_one = True

                                            path = f"{root}{self.w0_seed}/"
                                            seed_name = f"0-{second_cls}"
                                            img_path = f"{path}/{seed_name}.png"
                                            if not os.path.exists(img_path):
                                                os.makedirs(path, exist_ok=True)
                                                image.save(img_path)

                                                digit_info["l2_norm"] = img_l2
                                                with open(f"{path}/{seed_name}.json", 'w') as f:
                                                    (json.dump(digit_info, f, sort_keys=True, indent=4))

                                            found_mutation = True
                                            m_digit_info["accepted"] = m_accepted.tolist()
                                            m_digit_info["predicted-class"] = m_class.tolist()
                                            m_digit_info["exp-confidence"] = float(confidence)
                                            m_digit_info["predictions"] = m_predictions.tolist()
                                            m_digit_info["ssi"] = float(ssi)
                                            m_digit_info["l2_norm"] = m_img_l2
                                            m_digit_info["l2_distance"] = l2_distance

                                            m_path = f"{path}/{stylemix_cls}"
                                            m_name = f"/{int(l2_distance)}-{int(ssi * 100)}-{self.stylemix_seed}-{stylemix_cls}-{layer[0]}-{m_class}"
                                            os.makedirs(m_path, exist_ok=True)
                                            with open(f"{m_path}/{m_name}.json", 'w') as f:
                                                (json.dump(m_digit_info, f, sort_keys=True, indent=4))
                                            m_image.save(f"{m_path}/{m_name}.png")
                                    if idx == len(self.layers) and found_mutation:
                                        tried_all_layers = True
                                        break
                                self.stylemix_seed += 1
            self.w0_seed += self.step_size


def run_mimicry(class_idx, w0_seed=0, step_size=1):
    mimicry(class_idx=class_idx, w0_seed=w0_seed, step_size=step_size).search()

if __name__ == "__main__":
    run_mimicry(class_idx=9)
