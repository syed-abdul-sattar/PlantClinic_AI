
import torch
import timm
import json
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.nn.functional as F
import time


class PlantClinicSystem:

    def __init__(self, model_path, mapping_path, upsampler, leaf_model):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.upsampler = upsampler
        self.leaf_model = leaf_model

        with open(mapping_path, 'r') as f:
            self.class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.model = timm.create_model(
            'swin_large_patch4_window12_384',
            pretrained=False,
            num_classes=len(self.class_to_idx)
        )

        state_dict = torch.load(model_path, map_location=self.device)

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("head.fc", "head") if k.startswith("head.fc") else k
            new_state_dict[new_key] = v

        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((384,384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        self.leaf_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    # -------- LEAF CHECK --------
    def _is_leaf(self, img):
        try:
            x = self.leaf_transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.leaf_model(x)
            return torch.argmax(out).item() == 1
        except:
            return True   # fallback safety

    # -------- BIO GATE --------
    def _bio_gate(self, img_np):

        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        green_mask = cv2.inRange(hsv, np.array([25,40,40]), np.array([90,255,255]))
        green_ratio = np.sum(green_mask > 0)/(img_np.shape[0]*img_np.shape[1])

        edges = cv2.Canny(img_np,100,200)
        edge_density = np.sum(edges>0)/(img_np.shape[0]*img_np.shape[1])

        brightness = np.mean(hsv[:,:,2])

        return green_ratio > 0.05 and edge_density > 0.01 and brightness > 40

    # -------- CLAHE --------
    def _clahe(self, img_np):

        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l,a,b = cv2.split(lab)

        clahe = cv2.createCLAHE(3.0,(8,8))
        cl = clahe.apply(l)

        enhanced = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2RGB)

        blur = cv2.GaussianBlur(enhanced,(0,0),2)
        sharp = cv2.addWeighted(enhanced,1.5,blur,-0.5,0)

        return Image.fromarray(sharp)

    # -------- ESRGAN --------
    def _esrgan(self, img):
        img_np = np.array(img)
        output, _ = self.upsampler.enhance(img_np, outscale=4)
        return Image.fromarray(output)

    # -------- TTA --------
    def _predict(self, img):

        views = [
            img,
            TF.hflip(img),
            TF.adjust_brightness(img,1.2),
            TF.adjust_contrast(img,1.2),
            TF.rotate(img,10)
        ]

        batch = torch.stack([self.transform(v) for v in views]).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch)
            probs = F.softmax(outputs, dim=1)

        avg_probs = probs.mean(0)
        conf, pred = torch.max(avg_probs, dim=0)

        return avg_probs, pred.item(), conf.item()*100

    # -------- GRADCAM --------
    def _gradcam(self, img, cls):

        gradients, activations = [], []

        def f_hook(m,i,o): activations.append(o)
        def b_hook(m,gi,go): gradients.append(go[0])

        layer = self.model.layers[-1].blocks[-1].norm1

        h1 = layer.register_forward_hook(f_hook)
        h2 = layer.register_full_backward_hook(b_hook)

        x = self.transform(img).unsqueeze(0).to(self.device)

        out = self.model(x)
        self.model.zero_grad()
        out[0,cls].backward()

        grads = gradients[0][0].cpu().numpy()
        acts = activations[0][0].cpu().numpy()

        weights = np.mean(grads, axis=(0,1))

        cam = np.zeros(acts.shape[:2])

        for i,w in enumerate(weights):
            cam += w*acts[:,:,i]

        cam = np.maximum(cam,0)
        cam = cv2.resize(cam,(384,384))
        cam = (cam-cam.min())/(cam.max()+1e-8)

        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        img_np = cv2.resize(np.array(img),(384,384))

        h1.remove(); h2.remove()

        return cv2.addWeighted(img_np,0.6,heatmap,0.4,0)

    # -------- MAIN --------
    def run(self, path):

        start = time.time()

        try:
            img = Image.open(path).convert("RGB")
            img_np = np.array(img)
        except:
            return {"status":"ERROR"}

        if not self._is_leaf(img):
            return {"status":"REJECTED","message":"Not a plant"}

        if not self._bio_gate(img_np):
            return {"status":"REJECTED","message":"Invalid leaf"}

        blur = cv2.Laplacian(img_np,cv2.CV_64F).var()

        # HYBRID ENHANCEMENT
        if blur < 50:
            img_final = self._esrgan(img)
            mode = "ESRGAN"
        elif blur < 120:
            img_final = self._clahe(img_np)
            mode = "CLAHE"
        else:
            img_final = img
            mode = "NONE"

        probs, pred, conf = self._predict(img_final)

        # TOP 3
        top_vals, top_idx = torch.topk(probs, 3)
        top3 = [{"label": self.idx_to_class[top_idx[i].item()], "conf": top_vals[i].item()*100} for i in range(3)]

        # DYNAMIC THRESHOLD
        label = self.idx_to_class[pred].lower()

        if "healthy" in label:
            threshold = 90
        elif "blight" in label or "rot" in label:
            threshold = 92
        elif "virus" in label:
            threshold = 95
        else:
            threshold = 85

        status = "SUCCESS" if conf >= threshold else "UNCERTAIN"

        cam = self._gradcam(img_final, pred)
        path_cam = "/content/final_evidence.jpg"
        cv2.imwrite(path_cam, cv2.cvtColor(cam, cv2.COLOR_RGB2BGR))

        return {
            "status": status,
            "prediction": self.idx_to_class[pred],
            "confidence": round(conf,2),
            "top3": top3,
            "evidence": path_cam,
            "meta": {
                "enhancement": mode,
                "blur": blur,
                "time_sec": round(time.time()-start,2)
            }
        }
        