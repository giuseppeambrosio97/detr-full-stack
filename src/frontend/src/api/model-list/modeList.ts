import { apiRequest, RequestMethod } from "../apiRequest";
import { ModelType } from "./types";

export async function modelList(modelType: ModelType): Promise<string[]> {
    try {
        const res = await apiRequest<void, string[]>(RequestMethod.GET, `/model-list?modeltype=${modelType}`);
        return res;
    } catch (error) {
        throw error;
    }
}