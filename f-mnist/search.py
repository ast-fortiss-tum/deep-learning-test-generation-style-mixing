import os
import copy
import json
import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim 
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stylegan.renderer import Renderer
from config import STYLEGAN_INIT, SEARCH_LIMIT, STYLEMIX_SEED_LIMIT, INIT_PKL, STYLEMIX_LAYERS, FRONTIER_PAIRS
from predictor import Predictor
from utils import validate_mutation

Fine_Alpha = 0.0001

class mimicry:
    def __init__(self, class_idx=None, w0_seed=0, stylemix_seed=0, search_limit=SEARCH_LIMIT, step_size=1):
        self.state = STYLEGAN_INIT
        self.class_idx = class_idx
        self.w0_seed = w0_seed
        self.stylemix_seed = stylemix_seed
        self.search_limit = search_limit
        self.stylemix_seed_limit = STYLEMIX_SEED_LIMIT
        self.layers = STYLEMIX_LAYERS
        self.step_size = step_size
        self.state['renderer'] = Renderer()

    def render_state(self, state=None):
        if state is None:
            state = self.state
        state['renderer']._render_impl(
            res=state['generator_params'],
            pkl=INIT_PKL,
            w0_seeds=state['params']['w0_seeds'],
            class_idx=state['params']['class_idx'],
            mixclass_idx=state['params']['mixclass_idx'],
            stylemix_idx=state['params']['stylemix_idx'],
            stylemix_seed=state['params']['stylemix_seed'],
            img_normalize=state['params']['img_normalize'],
            to_pil=state['params']['to_pil'],
        )

        info = copy.deepcopy(state['params'])

        return state, info

    def save_difference_jetmap(self, img1, img2, save_path, pair_name):

        difference = np.abs(img1 - img2)
        
        plt.figure()
        plt.imshow(difference, cmap='jet', interpolation='none')
        plt.colorbar(label='Pixel Intensity Difference')
        
        # Calculate L2 distance and SSIM
        d_l2 = np.linalg.norm(difference)
        ssim_value = self.compute_ssim(img1, img2)
        
        plt.title(f'Difference Jetmap - D-L2: {d_l2:.2f}, SSIM: {ssim_value:.4f}')

        plt.savefig(f'{save_path}/{pair_name}_jetmap.png')
        plt.close()

    def compute_ssim(self, img1, img2):
        # Compute SSIM between two images
        data_range = img1.max() - img1.min() 
        ssim_value, _ = ssim(img1, img2, full=True, data_range=data_range)
        return ssim_value

    # Mask creation and interpolation methods
    def create_mask(self, image):

        # Convert image to grayscale if not already
        gray_image = np.array(image.convert('L'))
        threshold = 10  # Manual threshold value

        # Binary mask where pixels above threshold are considered foreground
        mask = (gray_image > threshold).astype(np.uint8)
        return mask

    def apply_mask(self, image, mask):
        
        return image * mask

    def interpolate_with_mask(self, img1, img2, alpha, mask1, mask2):
        # Apply masks to images before interpolation
        img1_masked = self.apply_mask(img1, mask1)
        img2_masked = self.apply_mask(img2, mask2)
        
        # Perform interpolation on masked images
        interpolated = img1_masked * (1 - alpha) + img2_masked * alpha
        
        # Combine masks for interpolated image
        combined_mask = np.logical_or(mask1, mask2).astype(np.uint8)
        interpolated = self.apply_mask(interpolated, combined_mask)
        
        return interpolated

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

            # Create mask for object
            mask = self.create_mask(image)

            accepted, confidence, predictions = Predictor().predict_datapoint(
                np.reshape(image_array, (-1, 28, 28, 1)),
                label
            )

            digit_info["accepted"] = accepted.tolist()
            digit_info["exp-confidence"] = float(confidence)
            digit_info["predictions"] = predictions.tolist()

            if accepted:
                found_at_least_one = False
                _, second_cls = np.argsort(-predictions)[:2]
                second_cls_confidence = predictions[second_cls]
                if second_cls_confidence:
                    for stylemix_cls, cls_confidence in enumerate(predictions):
                        if stylemix_cls != label and cls_confidence:
                            found_mutation = False
                            tried_all_layers = False

                            state["params"]["mixclass_idx"] = stylemix_cls
                            self.stylemix_seed = 0

                            while not found_mutation and not tried_all_layers and self.stylemix_seed < self.stylemix_seed_limit:

                                if self.stylemix_seed == self.w0_seed:
                                    self.stylemix_seed += 1
                                state["params"]["stylemix_seed"] = self.stylemix_seed

                                for idx, layer in enumerate(self.layers):
                                    state["params"]["stylemix_idx"] = layer

                                    m_digit, m_digit_info = self.render_state()
                                    m_image = m_digit['generator_params'].image
                                    m_image = m_image.crop((2, 2, m_image.width - 2, m_image.height - 2))
                                    m_image_array = np.array(m_image)

                                    # Create mask for the second image
                                    m_mask = self.create_mask(m_image)

                                    # Check confidence at alpha = 1
                                    alpha = 1.0
                                    interpolated_image = self.interpolate_with_mask(
                                        image_array, m_image_array, alpha=alpha, mask1=mask, mask2=m_mask
                                    )
                                    m_accepted, confidence, m_predictions = Predictor().predict_datapoint(
                                        np.reshape(interpolated_image, (-1, 28, 28, 1)),
                                        label
                                    )

                                    # If confidence is greater than 0.05 skip and start the next stylemixing
                                    if confidence > 0.05:
                                        print(f"Skipping stylemixing as confidence is {confidence:.4f} for alpha = 1")
                                        self.stylemix_seed += 1
                                        continue

                                    # Proceed with interpolation if confidence is less than 0.05
                                    print(f"Proceeding with interpolation as confidence is {confidence:.4f} for alpha = 1")

                                    # Initialize variables for tracking
                                    last_correct_image = None
                                    last_correct_alpha = None
                                    last_correct_predictions = None
                                    last_correct_confidence = None

                                    # Interpolate with initial steps
                                    alpha_values = np.linspace(0, 1, num=21)  # Steps of 0.05
                                    alpha_found_partial_confidence = False

                                    for alpha in alpha_values:
                                        # Perform interpolation with masking applied before interpolation
                                        interpolated_image = self.interpolate_with_mask(
                                            image_array, m_image_array, alpha=alpha, mask1=mask, mask2=m_mask
                                        )

                                        # Run classifier on interpolated image
                                        m_accepted, confidence, m_predictions = Predictor().predict_datapoint(
                                            np.reshape(interpolated_image, (-1, 28, 28, 1)),
                                            label
                                        )
                                        m_class = np.argmax(m_predictions)

                                        # Print classifier predictions
                                        print(f"Alpha: {alpha:.3f}, Accepted: {m_accepted}, Predicted Class: {m_class}, Confidence: {confidence}")

                                        # Save classifier outputs
                                        m_digit_info["alpha"] = alpha
                                        m_digit_info["accepted"] = m_accepted.tolist()
                                        m_digit_info["predicted-class"] = m_class.tolist()
                                        m_digit_info["exp-confidence"] = float(confidence)
                                        m_digit_info["predictions"] = m_predictions.tolist()

                                        if m_accepted:
                                            # image is still correctly classified
                                            last_correct_image = interpolated_image.copy()
                                            last_correct_alpha = alpha
                                            last_correct_predictions = m_predictions.copy()
                                            last_correct_confidence = confidence

                                            if confidence < 1 and not alpha_found_partial_confidence:
                                                # if confidence is less than 1, start finer interpolation
                                                alpha_found_partial_confidence = True
                                                fine_alpha_values = np.arange(alpha, 1.0001, Fine_Alpha)                                             


                                                for fine_alpha in fine_alpha_values:
                                                    interpolated_image = self.interpolate_with_mask(
                                                        image_array, m_image_array, alpha=fine_alpha, mask1=mask, mask2=m_mask
                                                    )

                                                    # Run classifier on interpolated image
                                                    m_accepted, confidence, m_predictions = Predictor().predict_datapoint(
                                                        np.reshape(interpolated_image, (-1, 28, 28, 1)),
                                                        label
                                                    )
                                                    m_class = np.argmax(m_predictions)

                                                    # Print classifier predictions
                                                    print(f"Fine Alpha: {fine_alpha:.3f}, Accepted: {m_accepted}, Predicted Class: {m_class}, Confidence: {confidence}")

                                                    # Save classifier outputs
                                                    m_digit_info["alpha"] = fine_alpha
                                                    m_digit_info["accepted"] = m_accepted.tolist()
                                                    m_digit_info["predicted-class"] = m_class.tolist()
                                                    m_digit_info["exp-confidence"] = float(confidence)
                                                    m_digit_info["predictions"] = m_predictions.tolist()

                                                    if m_accepted:
                                                        # Update last correct image
                                                        last_correct_image = interpolated_image.copy()
                                                        last_correct_alpha = fine_alpha
                                                        last_correct_predictions = m_predictions.copy()
                                                        last_correct_confidence = confidence
                                                    else:
                                                        # Classifier prediction has changed
                                                        if m_class == stylemix_cls:
                                                            valid_mutation, ssi, l2_distance, img_l2, m_img_l2 = validate_mutation(image_array, interpolated_image)

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
                                                                        json.dump(digit_info, f, sort_keys=True, indent=4)

                                                                found_mutation = True

                                                                # Save the last correctly classified image
                                                                if last_correct_image is not None:
                                                                    correct_img_uint8 = np.clip(last_correct_image, 0, 255).astype(np.uint8)
                                                                    correct_pil_image = Image.fromarray(correct_img_uint8)
                                                                    correct_img_name = f"{int(l2_distance)}-{int(ssi * 100)}-{self.stylemix_seed}-{stylemix_cls}-{layer[0]}-{last_correct_alpha:.3f}-correct.png"
                                                                    correct_img_path = f"{path}/{correct_img_name}"
                                                                    correct_pil_image.save(correct_img_path)

                                                                    # Save metadata for correct image
                                                                    correct_info = m_digit_info.copy()
                                                                    correct_info["alpha"] = last_correct_alpha
                                                                    correct_info["accepted"] = True
                                                                    correct_info["predictions"] = last_correct_predictions.tolist()
                                                                    correct_info["exp-confidence"] = float(last_correct_confidence)
                                                                    with open(f"{path}/{correct_img_name}.json", 'w') as f:
                                                                        json.dump(correct_info, f, sort_keys=True, indent=4)

                                                                # Save misclassified image
                                                                m_path = f"{path}/{stylemix_cls}"
                                                                m_name = f"{int(l2_distance)}-{int(ssi * 100)}-{self.stylemix_seed}-{stylemix_cls}-{layer[0]}-{fine_alpha:.3f}-misclassified.png"
                                                                os.makedirs(m_path, exist_ok=True)
                                                                m_digit_info["accepted"] = False
                                                                m_digit_info["predicted-class"] = m_class.tolist()
                                                                m_digit_info["exp-confidence"] = float(confidence)
                                                                m_digit_info["predictions"] = m_predictions.tolist()
                                                                m_digit_info["ssi"] = float(ssi)
                                                                m_digit_info["l2_norm"] = m_img_l2
                                                                m_digit_info["l2_distance"] = l2_distance
                                                                m_digit_info["alpha"] = fine_alpha
                                                                with open(f"{m_path}/{m_name}.json", 'w') as f:
                                                                    json.dump(m_digit_info, f, sort_keys=True, indent=4)
                                                                
                                                                # Convert and save interpolated image
                                                                interpolated_image_uint8 = np.clip(interpolated_image, 0, 255).astype(np.uint8)
                                                                interpolated_pil_image = Image.fromarray(interpolated_image_uint8)
                                                                interpolated_pil_image.save(f"{m_path}/{m_name}.png")

                                                                # Generate and save heatmap
                                                                self.save_difference_jetmap(image_array, interpolated_image, m_path, m_name)

                                                                
                                                                break
                                                        else:
                                                            break
                                                    if found_mutation:
                                                        break  # Break fine_alpha loop
                                                if found_mutation:
                                                    break  # Break alpha loop
                                        else:
                                            # Classifier prediction has changed
                                            if m_class == stylemix_cls:
                                                valid_mutation, ssi, l2_distance, img_l2, m_img_l2 = validate_mutation(image_array, interpolated_image)

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
                                                            json.dump(digit_info, f, sort_keys=True, indent=4)

                                                    found_mutation = True

                                                    # Save last correctly classified image
                                                    if last_correct_image is not None:
                                                        correct_img_uint8 = np.clip(last_correct_image, 0, 255).astype(np.uint8)
                                                        correct_pil_image = Image.fromarray(correct_img_uint8)
                                                        correct_img_name = f"{int(l2_distance)}-{int(ssi * 100)}-{self.stylemix_seed}-{stylemix_cls}-{layer[0]}-{last_correct_alpha:.3f}-correct.png"
                                                        correct_img_path = f"{path}/{correct_img_name}"
                                                        correct_pil_image.save(correct_img_path)

                                                        # Save metadata for correct image
                                                        correct_info = m_digit_info.copy()
                                                        correct_info["alpha"] = last_correct_alpha
                                                        correct_info["accepted"] = True
                                                        correct_info["predictions"] = last_correct_predictions.tolist()
                                                        correct_info["exp-confidence"] = float(last_correct_confidence)
                                                        with open(f"{path}/{correct_img_name}.json", 'w') as f:
                                                            json.dump(correct_info, f, sort_keys=True, indent=4)

                                                    # Save misclassified image
                                                    m_path = f"{path}/{stylemix_cls}"
                                                    m_name = f"{int(l2_distance)}-{int(ssi * 100)}-{self.stylemix_seed}-{stylemix_cls}-{layer[0]}-{alpha:.3f}-misclassified.png"
                                                    os.makedirs(m_path, exist_ok=True)
                                                    m_digit_info["accepted"] = False
                                                    m_digit_info["predicted-class"] = m_class.tolist()
                                                    m_digit_info["exp-confidence"] = float(confidence)
                                                    m_digit_info["predictions"] = m_predictions.tolist()
                                                    m_digit_info["ssi"] = float(ssi)
                                                    m_digit_info["l2_norm"] = m_img_l2
                                                    m_digit_info["l2_distance"] = l2_distance
                                                    m_digit_info["alpha"] = alpha
                                                    with open(f"{m_path}/{m_name}.json", 'w') as f:
                                                        json.dump(m_digit_info, f, sort_keys=True, indent=4)
                                                    
                                                    # Convert and save interpolated image
                                                    interpolated_image_uint8 = np.clip(interpolated_image, 0, 255).astype(np.uint8)
                                                    interpolated_pil_image = Image.fromarray(interpolated_image_uint8)
                                                    interpolated_pil_image.save(f"{m_path}/{m_name}.png")

                                                    # Generate and save heatmap
                                                    self.save_difference_jetmap(image_array, interpolated_image, m_path, m_name)

                                                    
                                                    break
                                        if found_mutation:
                                            break  # Break alpha loop
                                    if found_mutation:
                                        break  # Break layer loop
                                if found_mutation:
                                    break  # Break stylemix_seed loop
                                self.stylemix_seed += 1
            self.w0_seed += self.step_size

            
def run_mimicry(class_idx, w0_seed=0, step_size=1):
    mimicry_instance = mimicry(class_idx=class_idx, w0_seed=w0_seed, step_size=step_size)
    mimicry_instance.search()

if __name__ == "__main__":

    # run_mimicry(class_idx=9)
    # run_mimicry(class_idx=8)
    # run_mimicry(class_idx=7)
    run_mimicry(class_idx=6)
    # run_mimicry(class_idx=5)
    # run_mimicry(class_idx=4)
    # run_mimicry(class_idx=3)   
    # run_mimicry(class_idx=2)
    # run_mimicry(class_idx=1)
    # run_mimicry(class_idx=0)
