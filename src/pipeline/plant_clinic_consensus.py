# =========================================================
# IMPORTS
# =========================================================

import math
import uuid

from datetime import datetime

from collections import defaultdict


# =========================================================
# CONSENSUS SYSTEM
# =========================================================

class PlantClinicConsensusSystem:


    # =====================================================
    # INIT
    # =====================================================

    def __init__(self, base_system):

        self.base_system = base_system


    # =====================================================
    # LABEL FORMATTER
    # =====================================================

    def _format_label(self, label):

        return (

            label
            .replace("_", " ")
            .replace("(", "")
            .replace(")", "")

        )


    # =====================================================
    # ENTROPY CALCULATION
    # =====================================================

    def _calculate_entropy(self, probs):

        entropy = 0.0

        for p in probs:

            p = float(p)

            if p > 0:

                entropy -= p * math.log(p)

        return entropy


    # =====================================================
    # MULTI IMAGE PIPELINE
    # =====================================================

    def run_multi_image(self, image_paths):


        # =================================================
        # NO IMAGES
        # =================================================

        if len(image_paths) == 0:

            return {

                "status": "FAILED",

                "message":
                    "No plant images were uploaded for analysis.",

                "total_uploaded":
                    0

            }


        # =================================================
        # IMAGE LIMIT WARNING
        # =================================================

        image_limit_warning = None

        ignored_images = []

        if len(image_paths) > 3:

            ignored_images = list(

                range(
                    4,
                    len(image_paths) + 1
                )

            )

            image_limit_warning = (

                f"{len(image_paths)} images were uploaded. "
                f"PlantClinic AI currently analyzes "
                f"a maximum of 3 images per session "
                f"to maintain reliable multi-image "
                f"consensus diagnosis. "

                f"Images used for analysis: [1, 2, 3]. "

                f"Ignored images: {ignored_images}."

            )


        # =================================================
        # LIMIT IMAGES
        # =================================================

        original_uploaded_count = len(image_paths)

        image_paths = image_paths[:3]


        # =================================================
        # SINGLE IMAGE WARNING
        # =================================================

        single_image_warning = None

        if len(image_paths) == 1:

            single_image_warning = (

                "Only one image was available "
                "for diagnosis. Multi-image "
                "consensus reliability may "
                "be reduced."

            )

            print(

                "\nWARNING:"
            )

            print(
                single_image_warning
            )


        # =================================================
        # STORE RESULTS
        # =================================================

        image_results = []


        # =================================================
        # RUN PIPELINE
        # =================================================

        for idx, path in enumerate(image_paths):

            print(
                f"\nPROCESSING IMAGE {idx + 1}"
            )

            result = self.base_system.run(path)

            image_results.append({

                "original_index":
                    idx + 1,

                "result":
                    result

            })


        # =================================================
        # FILTER VALID RESULTS
        # =================================================

        rejected_images = []

        valid_results = []


        for item in image_results:

            result = item["result"]


            # =============================================
            # STATUS CHECK
            # =============================================

            if result["status"] not in [

                "SUCCESS",
                "UNCERTAIN"

            ]:

                rejection_reason = result["status"]


                # =========================================
                # FRIENDLY REJECTION REASONS
                # =========================================

                if rejection_reason == "NON_LEAF":

                    rejection_reason = (

                        "The uploaded image does not "
                        "appear to contain a clear "
                        "plant leaf."

                    )

                elif rejection_reason == "FAILED":

                    rejection_reason = (

                        "Image processing failed due "
                        "to insufficient visual quality."

                    )

                elif rejection_reason == "REJECTED":

                    rejection_reason = (

                        "The uploaded image could not "
                        "be validated as a plant image."

                    )

                else:

                    rejection_reason = (

                        "The uploaded image could not "
                        "be analyzed reliably."

                    )


                rejected_images.append({

                    "image":
                        item["original_index"],

                    "reason":
                        rejection_reason

                })


                print(

                    f"\nIMAGE {item['original_index']} EXCLUDED"

                )

                print(
                    f"Reason: {rejection_reason}"
                )

                continue


            # =============================================
            # ENTROPY CHECK
            # =============================================

            probs = result["raw_probs"]

            entropy = self._calculate_entropy(

                probs

            )


            # =============================================
            # HARD UNCERTAINTY REJECTION
            # =============================================

            if entropy > 1.5:

                uncertainty_reason = (

                    "The diagnostic evidence in this "
                    "image was highly uncertain."

                )

                rejected_images.append({

                    "image":
                        item["original_index"],

                    "reason":
                        uncertainty_reason

                })

                print(

                    f"\nIMAGE {item['original_index']} EXCLUDED"

                )

                print(
                    f"Reason: {uncertainty_reason}"
                )

                continue


            # =============================================
            # VALID RESULT
            # =============================================

            valid_results.append({

                "original_index":
                    item["original_index"],

                "result":
                    result,

                "entropy":
                    entropy

            })


        # =================================================
        # ALL REJECTED
        # =================================================

        if len(valid_results) == 0:

            return {

                "status": "REJECTED",

                "message":

                    "All uploaded images were "
                    "excluded because reliable "
                    "plant disease evidence "
                    "could not be detected.",

                "total_uploaded":
                    original_uploaded_count,

                "rejected_images_info":
                    rejected_images,

                "image_limit_warning":
                    image_limit_warning

            }


        # =================================================
        # DISEASE SCORES
        # =================================================

        disease_scores = defaultdict(float)


        # =================================================
        # IMAGE SUMMARIES
        # =================================================

        per_image_summary = []


        # =================================================
        # ACCUMULATE EVIDENCE
        # =================================================

        for item in valid_results:

            result = item["result"]

            pred = result["prediction"]

            confidence = result["confidence"]


            # =============================================
            # CONFIDENCE WEIGHTED VOTING
            # =============================================

            disease_scores[pred] += confidence


            # =============================================
            # IMAGE SUMMARY
            # =============================================

            per_image_summary.append({

                "image":
                    item["original_index"],

                "prediction":
                    pred,

                "confidence":
                    round(confidence, 2),

                "entropy":
                    round(item["entropy"], 3)

            })


        # =================================================
        # FINAL PREDICTION
        # =================================================

        final_prediction = max(

            disease_scores,

            key=disease_scores.get

        )


        # =================================================
        # AGREEMENT RATIO
        # =================================================

        total_score = sum(

            disease_scores.values()

        )

        final_score = disease_scores[

            final_prediction

        ]


        agreement_ratio = (

            final_score / total_score

        )


        # =================================================
        # FINAL CONFIDENCE
        # =================================================

        supporting_confidences = [

            item["result"]["confidence"]

            for item in valid_results

            if item["result"]["prediction"]

               == final_prediction

        ]


        avg_model_confidence = (

            sum(supporting_confidences)

            / len(supporting_confidences)

        )


        final_confidence = (

            avg_model_confidence

            * agreement_ratio

        )


        # =================================================
        # CONSENSUS STRENGTH
        # =================================================

        if agreement_ratio >= 0.80:

            consensus_strength = "HIGH"

        elif agreement_ratio >= 0.60:

            consensus_strength = "MODERATE"

        else:

            consensus_strength = "LOW"


        # =================================================
        # UNCERTAINTY FLAG
        # =================================================

        uncertain = False

        if consensus_strength == "LOW":

            uncertain = True


        # =================================================
        # TRUST MESSAGE
        # =================================================

        if consensus_strength == "HIGH":

            consensus_message = (

                "Multiple uploaded images "
                "consistently support the "
                "same plant disease diagnosis."

            )

        elif consensus_strength == "MODERATE":

            consensus_message = (

                "Most uploaded images support "
                "the same diagnosis, although "
                "some variation exists between images."

            )

        else:

            consensus_message = (

                "Uploaded images produced "
                "conflicting diagnostic evidence. "
                "Please upload clearer images "
                "showing visible symptoms."

            )


        # =================================================
        # SINGLE VALID IMAGE WARNING
        # =================================================

        if (

            len(valid_results) == 1

            and

            len(image_paths) > 1

        ):

            consensus_message += (

                " Only one image contained "
                "sufficient diagnostic evidence "
                "for reliable analysis."

            )


        # =================================================
        # IMAGE LIMIT WARNING
        # =================================================

        if image_limit_warning is not None:

            consensus_message += (

                " " + image_limit_warning

            )


        # =================================================
        # HEALTHY / DISEASE CONFLICT
        # =================================================

        healthy_present = any(

            "healthy" in
            item["result"]["prediction"].lower()

            for item in valid_results

        )


        disease_present = any(

            "healthy" not in
            item["result"]["prediction"].lower()

            for item in valid_results

        )


        if healthy_present and disease_present:

            uncertain = True

            consensus_message += (

                " Both healthy and diseased "
                "visual patterns were detected "
                "across uploaded images."

            )


        # =================================================
        # OUTLIER IMAGES
        # =================================================

        outlier_images = []


        for item in valid_results:

            if (

                item["result"]["prediction"]

                != final_prediction

            ):

                outlier_images.append(

                    item["original_index"]

                )


        # =================================================
        # OUTLIER MESSAGE
        # =================================================

        if len(outlier_images) > 0:

            outlier_message = (

                "Some uploaded images produced "
                "different diagnostic patterns "
                "from the final consensus diagnosis."

            )

        else:

            outlier_message = (

                "All analyzed images supported "
                "the same diagnosis."

            )


        # =================================================
        # SEVERITY AGGREGATION
        # =================================================

        severity_map = {

            "MILD": 1,
            "MODERATE": 2,
            "SEVERE": 3

        }


        reverse_map = {

            1: "MILD",
            2: "MODERATE",
            3: "SEVERE"

        }


        severity_scores = [

            severity_map[

                item["result"]["severity"]

            ]

            for item in valid_results

            if item["result"]["prediction"]

               == final_prediction

        ]


        avg_severity = round(

            sum(severity_scores)

            / len(severity_scores)

        )


        severity = reverse_map[

            avg_severity

        ]


        # =================================================
        # FINAL RESULT
        # =================================================

        return {

            "status":
                "SUCCESS",

            "final_prediction":
                final_prediction,

            "final_confidence":
                round(final_confidence, 2),

            "consensus_strength":
                consensus_strength,

            "uncertain":
                uncertain,

            "severity":
                severity,

            "message":
                consensus_message,

            "valid_images":
                len(valid_results),

            "total_uploaded":
                original_uploaded_count,

            "outlier_images":
                outlier_images,

            "outlier_message":
                outlier_message,

            "per_image_summary":
                per_image_summary,

            "rejected_images_info":
                rejected_images,

            "image_limit_warning":
                image_limit_warning,

            "single_image_warning":
                single_image_warning,

            "_debug": {

                "agreement_ratio":
                    round(agreement_ratio, 4),

                "disease_scores":
                    dict(disease_scores)

            }

        }


    # =====================================================
    # DISPLAY FINAL RESULT
    # =====================================================

    def display_result(self, result):


        # =================================================
        # REPORT INFO
        # =================================================

        report_id = str(uuid.uuid4())[:8].upper()

        timestamp = datetime.now().strftime(

            "%Y-%m-%d %H:%M:%S"

        )


        print(
            "\n=================================================="
        )

        print(
            "FINAL DIAGNOSIS REPORT"
        )

        print(
            "=================================================="
        )

        print(
            f"Report ID : {report_id}"
        )

        print(
            f"Generated : {timestamp}"
        )


        # =================================================
        # STATUS
        # =================================================

        print(
            "\nSTATUS :",
            result["status"]
        )


        # =================================================
        # SUCCESS CASE
        # =================================================

        if result["status"] == "SUCCESS":


            # =============================================
            # FINAL DIAGNOSIS
            # =============================================

            print(
                "\nFINAL DIAGNOSIS"
            )

            print(
                "--------------------------------------------------"
            )


            print(
                "Predicted Disease :",
                self._format_label(

                    result["final_prediction"]

                )
            )


            print(
                "Confidence Score  :",
                f'{result["final_confidence"]:.2f}%'
            )


            print(
                "Severity Level    :",
                result["severity"]
            )


            # =============================================
            # CONFIDENCE INTERPRETATION
            # =============================================

            print(
                "\nCONFIDENCE INTERPRETATION"
            )

            print(
                "--------------------------------------------------"
            )


            confidence = result["final_confidence"]


            if confidence >= 95:

                interpretation = (

                    "Very strong diagnostic confidence."

                )

            elif confidence >= 85:

                interpretation = (

                    "Strong diagnostic confidence."

                )

            elif confidence >= 70:

                interpretation = (

                    "Moderate diagnostic confidence."

                )

            else:

                interpretation = (

                    "Limited diagnostic confidence."

                )


            print(
                interpretation
            )


            # =============================================
            # RELIABILITY
            # =============================================

            print(
                "\nDIAGNOSTIC RELIABILITY"
            )

            print(
                "--------------------------------------------------"
            )

            print(
                "Consensus Strength :",
                result["consensus_strength"]
            )


            if result["consensus_strength"] == "HIGH":

                print(
                    "Reliability        : High diagnostic reliability"
                )

            elif result["consensus_strength"] == "MODERATE":

                print(
                    "Reliability        : Moderate diagnostic reliability"
                )

            else:

                print(
                    "Reliability        : Low diagnostic reliability"
                )


            # =============================================
            # SYSTEM ANALYSIS
            # =============================================

            print(
                "\nSYSTEM ANALYSIS"
            )

            print(
                "--------------------------------------------------"
            )

            print(
                result["message"]
            )


            # =============================================
            # SINGLE IMAGE WARNING
            # =============================================

            if result["single_image_warning"] is not None:

                print(
                    "\nSINGLE IMAGE NOTICE"
                )

                print(
                    "--------------------------------------------------"
                )

                print(
                    result["single_image_warning"]
                )


            # =============================================
            # IMAGE LIMIT WARNING
            # =============================================

            if result["image_limit_warning"] is not None:

                print(
                    "\nIMAGE LIMIT NOTICE"
                )

                print(
                    "--------------------------------------------------"
                )

                print(
                    result["image_limit_warning"]
                )


            # =============================================
            # IMAGE SUMMARY
            # =============================================

            print(
                "\nIMAGE PROCESSING SUMMARY"
            )

            print(
                "--------------------------------------------------"
            )

            print(
                "Total Images Uploaded        :",
                result["total_uploaded"]
            )

            print(
                "Successfully Analyzed Images :",
                result["valid_images"]
            )

            print(
                "Excluded Images              :",
                len(result["rejected_images_info"])
            )


            # =============================================
            # REJECTED IMAGES
            # =============================================

            if len(result["rejected_images_info"]) > 0:

                print(
                    "\nEXCLUDED IMAGE DETAILS"
                )

                print(
                    "--------------------------------------------------"
                )

                for item in result["rejected_images_info"]:

                    print(
                        f"\nImage {item['image']}"
                    )

                    print(
                        "Reason :",
                        item["reason"]
                    )


            # =============================================
            # CONSENSUS ANALYSIS
            # =============================================

            print(
                "\nCONSENSUS ANALYSIS"
            )

            print(
                "--------------------------------------------------"
            )

            print(
                result["outlier_message"]
            )


            if len(result["outlier_images"]) > 0:

                print(
                    "Outlier Images :",
                    result["outlier_images"]
                )


            # =============================================
            # PER IMAGE ANALYSIS
            # =============================================

            print(
                "\nPER-IMAGE ANALYSIS"
            )

            print(
                "=================================================="
            )


            for item in result["per_image_summary"]:

                print(
                    f"\nImage {item['image']}"
                )

                print(
                    "--------------------------------------------------"
                )

                print(
                    "Prediction        :",
                    self._format_label(

                        item["prediction"]

                    )
                )

                print(
                    "Confidence        :",
                    f"{item['confidence']}%"
                )


                # =========================================
                # EVIDENCE QUALITY
                # =========================================

                entropy = item["entropy"]


                if entropy < 0.10:

                    quality = "Excellent"

                elif entropy < 0.30:

                    quality = "Good"

                else:

                    quality = "Moderate"


                print(
                    "Evidence Quality  :",
                    quality
                )


            # =============================================
            # UNCERTAINTY WARNING
            # =============================================

            if result["uncertain"]:

                print(
                    "\nDIAGNOSTIC WARNING"
                )

                print(
                    "--------------------------------------------------"
                )

                print(
                    "Conflicting or inconsistent "
                    "visual evidence reduced "
                    "diagnostic certainty."
                )


            # =============================================
            # RECOMMENDATION
            # =============================================

            print(
                "\nRECOMMENDATION"
            )

            print(
                "=================================================="
            )

            print(

                f"The uploaded plant images "

                f"indicate "

                f"{self._format_label(result['final_prediction'])} "

                f"with "

                f"{result['consensus_strength'].lower()} "

                f"diagnostic agreement."

            )


            print(
                "\nYou may proceed with "
                "plant disease management "
                "and agricultural treatment planning."
            )


            # =============================================
            # IMAGE QUALITY GUIDANCE
            # =============================================

            if len(result["rejected_images_info"]) > 0:

                print(
                    "\nIMAGE QUALITY GUIDANCE"
                )

                print(
                    "--------------------------------------------------"
                )

                print(
                    "For best diagnostic accuracy:"
                )

                print(
                    "• Use natural lighting"
                )

                print(
                    "• Keep the leaf centered"
                )

                print(
                    "• Avoid blurry motion"
                )

                print(
                    "• Capture visible symptoms clearly"
                )

                print(
                    "• Upload multiple plant views when possible"
                )


            # =============================================
            # DISCLAIMER
            # =============================================

            print(
                "\nDISCLAIMER"
            )

            print(
                "=================================================="
            )

            print(
                "PlantClinic AI provides "
                "AI-assisted agricultural analysis "
                "and should support — not replace — "
                "professional agricultural consultation."
            )


        # =================================================
        # FAILED / REJECTED CASE
        # =================================================

        else:

            print(
                "\nANALYSIS RESULT"
            )

            print(
                "--------------------------------------------------"
            )

            print(
                result["message"]
            )


            # =============================================
            # IMAGE LIMIT WARNING
            # =============================================

            if (

                "image_limit_warning"

                in result

                and

                result["image_limit_warning"] is not None

            ):

                print(
                    "\nIMAGE LIMIT NOTICE"
                )

                print(
                    "--------------------------------------------------"
                )

                print(
                    result["image_limit_warning"]
                )


            # =============================================
            # REJECTED IMAGE DETAILS
            # =============================================

            if (

                "rejected_images_info"

                in result

                and

                len(result["rejected_images_info"]) > 0

            ):

                print(
                    "\nEXCLUDED IMAGE DETAILS"
                )

                print(
                    "--------------------------------------------------"
                )

                for item in result["rejected_images_info"]:

                    print(
                        f"\nImage {item['image']}"
                    )

                    print(
                        "Reason :",
                        item["reason"]
                    )


            print(
                "\nPlease upload clearer plant images "
                "showing visible disease symptoms "
                "for reliable diagnosis."
            )


        # =================================================
        # END
        # =================================================

        print(
            "\n=================================================="
        )

        print(
            "END OF REPORT"
        )

        print(
            "==================================================\n"
        )