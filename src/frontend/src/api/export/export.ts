import { apiRequest, RequestMethod } from "../apiRequest";
import { ExportResponse } from "./types";


export async function exportModel(modelName: string): Promise<ExportResponse> {
    try {
        const res = await apiRequest<void, ExportResponse>(RequestMethod.GET, `/export?model_name=${modelName}`);
        return res;
    } catch (error) {
        throw error;
    }
}