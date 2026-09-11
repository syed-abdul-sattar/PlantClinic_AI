import torch
import timm
import json
import cv2
import math
from collections import defaultdict
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn

from torchvision.models import mobilenet_v3_small
import time


class PlantClinicSystem:

    def __init__(self,
                 disease_model_path,
                 class_mapping_path,
                 leaf_model_path,
                 upsampler=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -------------------------
        # CLASS MAPPING
        # -------------------------
        with open(class_mapping_path, "r") as f:
            self.class_to_idx = json.load(f)

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # -------------------------
        # DISEASE MODEL
        # -------------------------
        self.model = timm.create_model(
            "swin_large_patch4_window12_384",
            pretrained=False,
            num_classes=len(self.class_to_idx)
        )

        state_dict = torch.load(disease_model_path, map_location=self.device)

        fixed_state_dict = {}

        for k, v in state_dict.items():

            if k == "head.weight":
                new_key = "head.fc.weight"

            elif k == "head.bias":
                new_key = "head.fc.bias"

            else:
                new_key = k

            fixed_state_dict[new_key] = v

        self.model.load_state_dict(fixed_state_dict)

        self.model.to(self.device).eval()

        # -------------------------
        # LEAF CLASSIFIER
        # -------------------------
        print("LOADING LEAF GATEKEEPER MODEL")

        self.leaf_model = mobilenet_v3_small(
          weights=None
)

        self.leaf_model.classifier[3] = nn.Linear(
          self.leaf_model.classifier[3].in_features,
          2
)

        state_dict = torch.load(
          leaf_model_path,
          map_location=self.device
)

        self.leaf_model.load_state_dict(state_dict)

        self.leaf_model.to(self.device)

        self.leaf_model.eval()
        self.leaf_classes = {

          0: "leaf",

          1: "non_leaf"
}

        print("LEAF GATEKEEPER LOADED SUCCESSFULLY")
        

        # -------------------------
        # ESRGAN
        # -------------------------
        self.upsampler = upsampler

        # -------------------------
        # TRANSFORMS
        # -------------------------
        self.transform = transforms.Compose([
            transforms.Resize((384,384)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485,0.456,0.406],
                [0.229,0.224,0.225]
            )
        ])

        self.leaf_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485,0.456,0.406],
                [0.229,0.224,0.225]
            )
        ])


    # ------------------------------------------------
    # LEAF DETECTION
    # ------------------------------------------------

    def _leaf_score(self, img):

        try:
          
          x = self.leaf_transform(img).unsqueeze(0).to(self.device)

          with torch.no_grad():
              out = self.leaf_model(x)

              prob = torch.softmax(out, dim=1)
          leaf_prob = prob[0][0].item()

          non_leaf_prob = prob[0][1].item()
          print(

            f"Leaf: {leaf_prob:.4f} | "
            f"NonLeaf: {non_leaf_prob:.4f}"
        )


          return leaf_prob, non_leaf_prob


        except Exception as e:

          print(f"Leaf Score Error : {e}")

          return 0.5,0.5
           

# =========================================================
# ENTROPY CALCULATION
# =========================================================

    def _calculate_entropy(self, probs):

      entropy = 0.0

      for p in probs:

        p = float(p)

        if p > 0:

          entropy -= p * math.log(p)

      return entropy
    # =========================================================
# IMAGE QUALITY SCORE
# =========================================================

    def _quality_weight(self, blur_score):

      # blur_score already exists in pipeline
      # normalize into soft quality weight

      if blur_score >= 120:
        return 1.0

      elif blur_score >= 80:
        return 0.9

      elif blur_score >= 50:
        return 0.75

      elif blur_score >= 30:
        return 0.60

      else:
        return 0.45
    # ------------------------------------------------
    # BIO SCORE
    # ------------------------------------------------

    

    # ------------------------------------------------
    # CLAHE ENHANCEMENT
    # ------------------------------------------------

    def _clahe(self, img_np):

        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

        l,a,b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))

        cl = clahe.apply(l)

        merged = cv2.merge((cl,a,b))

        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        blur = cv2.GaussianBlur(enhanced,(0,0),2)

        sharp = cv2.addWeighted(enhanced,1.5,blur,-0.5,0)

        return Image.fromarray(sharp)


    # ------------------------------------------------
    # ESRGAN
    # ------------------------------------------------

    def _esrgan(self, img):

        img_np = np.array(img)

        output,_ = self.upsampler.enhance(img_np,outscale=4)

        return Image.fromarray(output)


    # ------------------------------------------------
    # TTA PREDICTION (Reduced)
    # ------------------------------------------------

    def _predict(self,img):

        views = [
            img,
            TF.hflip(img),
            TF.adjust_brightness(img,1.1)
        ]

        batch = torch.stack(
            [self.transform(v) for v in views]
        ).to(self.device)

        with torch.no_grad():

            outputs = self.model(batch)

            probs = F.softmax(outputs,dim=1)

        avg_probs = probs.mean(0)

        conf,pred = torch.max(avg_probs,dim=0)

        return avg_probs,pred.item(),conf.item()*100


    # ------------------------------------------------
    # GRADCAM
    # ------------------------------------------------

    def _gradcam(self,img,cls):

        gradients=[]
        activations=[]

        def forward_hook(module,input,output):
            activations.append(output)

        def backward_hook(module,grad_in,grad_out):
            gradients.append(grad_out[0])

        layer=self.model.layers[-1].blocks[-1].norm1

        h1=layer.register_forward_hook(forward_hook)
        h2=layer.register_full_backward_hook(backward_hook)

        x=self.transform(img).unsqueeze(0).to(self.device)

        output=self.model(x)

        self.model.zero_grad()

        output[0,cls].backward()

        grads=gradients[0][0].detach().cpu().numpy()
        acts=activations[0][0].detach().cpu().numpy()

        weights=np.mean(grads,axis=(0,1))

        cam=np.zeros(acts.shape[:2])

        for i,w in enumerate(weights):
            cam+=w*acts[:,:,i]

        cam=np.maximum(cam,0)

        cam=cv2.resize(cam,(384,384))

        cam=(cam-cam.min())/(cam.max()+1e-8)

        heatmap=cv2.applyColorMap(
            np.uint8(255*cam),
            cv2.COLORMAP_JET
        )

        img_np=cv2.resize(np.array(img),(384,384))

        result=cv2.addWeighted(img_np,0.6,heatmap,0.4,0)

        h1.remove()
        h2.remove()

        return result,cam


    # ------------------------------------------------
    # MAIN PIPELINE
    # ------------------------------------------------

    def run(self,image_path):

        start_time=time.time()

        try:

            img=Image.open(image_path).convert("RGB")

            img_np=np.array(img)

        except Exception as e:

            return {
                "status":"ERROR",
                "message":str(e)
            }


        # ---------- LEAF CHECK ----------

        leaf_score, non_leaf_score = self._leaf_score(img)
        print("Leaf Score:", leaf_score)
        if leaf_score<0.30:

            return {

              "status":"REJECTED",

              "message":
              "Uploaded image is not recognized as a plant leaf. "
              "Please upload a clear leaf image for disease analysis.",

              "leaf_confidence":
              round(leaf_score*100,2),

              "non_leaf_confidence":
              round(non_leaf_score*100,2)
}
        


        # ---------- BLUR DETECTION ----------

        blur=cv2.Laplacian(img_np,cv2.CV_64F).var()


        # ---------- ENHANCEMENT ----------

        if blur<50 and self.upsampler is not None:

            img_final=self._esrgan(img)
            mode="ESRGAN"

        elif blur<120:

            img_final=self._clahe(img_np)
            mode="CLAHE"

        else:

            img_final=img
            mode="NONE"


        # ---------- PREDICTION ----------

        probs,pred,conf=self._predict(img_final)
        raw_probs = probs.cpu().numpy().tolist()


        # ---------- OOD DETECTION ----------

        if conf<40:

            return {
                "status":"UNKNOWN",
                "message":"Disease not recognized in trained dataset"
            }


        # ---------- TOP3 ----------

        top_vals,top_idx=torch.topk(probs,3)

        top3=[]

        for i in range(3):

            top3.append({

                "label":self.idx_to_class[top_idx[i].item()],
                "confidence":round(top_vals[i].item()*100,2)

            })


        label=self.idx_to_class[pred].lower()


        # ---------- DYNAMIC THRESHOLD ----------

        if "healthy" in label:

            threshold=90

        elif "virus" in label:

            threshold=95

        elif "blight" in label or "rot" in label:

            threshold=92

        else:

            threshold=85


        if conf>=threshold:

            status="SUCCESS"

        elif conf>=40:

            status="UNCERTAIN"

        else:

            status="UNKNOWN"


        # ---------- GRADCAM ----------

        cam_img,cam_map=self._gradcam(img_final,pred)


        # ---------- SEVERITY ESTIMATION ----------

        disease_pixels=np.sum(cam_map>0.6)

        total_pixels=cam_map.shape[0]*cam_map.shape[1]

        severity_ratio=disease_pixels/total_pixels

        if severity_ratio<0.05:

            severity="MILD"

        elif severity_ratio<0.20:

            severity="MODERATE"

        else:

            severity="SEVERE"


        evidence_path="/content/evidence.jpg"

        cv2.imwrite(
            evidence_path,
            cv2.cvtColor(cam_img,cv2.COLOR_RGB2BGR)
        )


        return {

            "status":status,
            "prediction":self.idx_to_class[pred],
            "confidence":round(conf,2),
            "severity":severity,
            "top3":top3,
            "raw_probs": raw_probs,

            "blur_score": float(round(blur,2)),
            "evidence":evidence_path,

            "meta":{

                "enhancement":mode,
                "processing_time":round(time.time()-start_time,2)

            }

        }