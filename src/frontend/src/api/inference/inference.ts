import { apiRequest, RequestMethod } from "../apiRequest";
import { TInferenceRequest, TInferenceResponse } from "./types";


export async function inference(request: TInferenceRequest): Promise<TInferenceResponse> {
    try {
        const res = await apiRequest<TInferenceRequest, TInferenceResponse>(RequestMethod.POST, "/inference", request);
        return res;
    } catch (error) {
        throw error;
    }
}

export async function inferenceExamples(): Promise<string[]> {
    try {
        const res = await apiRequest<void, string[]>(RequestMethod.GET, "/inference/examples");
        return res;
    } catch (error) {
        throw error;
    }
}