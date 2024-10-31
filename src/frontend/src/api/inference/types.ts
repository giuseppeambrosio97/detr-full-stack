
type TInferenceRequest = {
    model_name: string;
    confidence: number;
    image_base64: string;
};

type TInferenceResponse = {
    annotated_image_base64: string;
    metrics: Record<string, number>;
};

export type {
    TInferenceRequest,
    TInferenceResponse,
}