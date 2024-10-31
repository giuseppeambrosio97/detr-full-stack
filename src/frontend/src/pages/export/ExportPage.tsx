import { exportModel, ExportResponse } from '@/api/export';
import ModelDropdown from '@/components/ModelDropdown';
import config from '@/config/config';
import { Button } from 'primereact/button';
import { ProgressBar } from 'primereact/progressbar';
import { Toast } from 'primereact/toast';
import { useRef, useState } from 'react';

function ExportPage() {
    const [modelName, setModelName] = useState<string>('detr_simple_demo');
    const [exportedResponse, setExportedResponse] = useState<ExportResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const toastRef = useRef<Toast | null>(null);

    const exportW = async () => {
        try {
            setLoading(true);
            const response = await exportModel(modelName);
            setExportedResponse(response);
            toastRef.current?.show({
                severity: 'success',
                detail: 'Export Success.',
            });
        } catch (error) {
            toastRef.current?.show({
                severity: 'error',
                detail: 'Export Failed.',
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="w-full h-full px-8 py-2 flex flex-col items-center">
            <Toast ref={toastRef} />
            <h2 className="text-center text-2xl mb-4">Model Export ONNX</h2>

            <div className={`grid grid-cols-${exportedResponse !== null ? '2' : '1'} gap-8`}>
                <div className="flex flex-col gap-2">
                    <span>Model to export:</span>
                    <ModelDropdown
                        modelName={modelName}
                        onChange={setModelName}
                        modelType="exportable"
                    />
                    {loading && (
                        <ProgressBar
                            mode="indeterminate"
                            style={{ height: '3px', padding: '3px' }}
                        />
                    )}
                    <Button
                        label="Export"
                        className="shadow p-2 bg-primary text-white"
                        onClick={exportW}
                        disabled={loading}
                    />
                </div>

                {exportedResponse !== null && (
                    <div className="flex flex-col gap-4 items-start">
                        <div>
                            <h4 className="font-semibold">File Details:</h4>
                            <ul className="text-sm">
                                <li className="flex gap-2 items-center">
                                    <span>File Name: </span>
                                    <span>{exportedResponse.file_name}</span>
                                    <Button
                                        icon="pi pi-download"
                                        tooltip="Download file"
                                        className="shadow p-2 bg-primary text-white w-8 h-8 ml-1"
                                        onClick={() =>
                                            window.open(
                                                `${config.backend.baseUrl}/exported-model/${exportedResponse.file_name}`,
                                                '_blank'
                                            )
                                        }
                                    />
                                </li>
                                <li>
                                    <span>File Size: </span>
                                    <span>{exportedResponse.file_size}MB</span>
                                </li>
                            </ul>
                        </div>

                        <div className="mt-4 w-60">
                            <h4 className="font-semibold">Metrics:</h4>
                            <ul className="text-sm">
                                {exportedResponse.metrics &&
                                    Object.entries(exportedResponse.metrics).map(([key, value]) => (
                                        <li key={key} className="flex justify-between">
                                            <span>{key}:</span>
                                            <span>{value.toFixed(2)}ms</span>
                                        </li>
                                    ))}
                            </ul>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default ExportPage;
