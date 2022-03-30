from ml.data import process_data
from ml.model import inference, compute_model_metrics


def calculate_education_slice_metrics(
    model, encoder, lb, test_data, categorical_features
):
    with open("../metrics/slice_output.txt", mode="w+") as ouput:
        ouput.write("Model metrics per education class slice\n\n")
        for cls in test_data["education"].unique():
            df_temp = test_data[test_data["education"] == cls]
            X_sliced, y_sliced, encoder, lb = process_data(
                df_temp,
                categorical_features=categorical_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb,
            )
            prediction = inference(model, X_sliced)
            precision, recall, fbeta = compute_model_metrics(y_sliced, prediction)

            ouput.write(f"Class: {cls}")
            ouput.write("\n")
            ouput.write(f"Precision: {precision}, Recall: {recall}, Fbeta: {fbeta}")
            ouput.write("\n\n")
