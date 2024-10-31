import React, { useEffect, useRef, useState } from 'react';
import {
    inference,
    inferenceExamples,
    TInferenceRequest,
    TInferenceResponse,
} from '@/api/inference';
import { Button } from 'primereact/button';
import { InputNumber } from 'primereact/inputnumber';
import { Image } from 'primereact/image';
import { ProgressSpinner } from 'primereact/progressspinner';
import { Dropdown } from 'primereact/dropdown';
import { Messages } from 'primereact/messages';
import config from '@/config/config';
import { Tooltip } from 'primereact/tooltip';
import { Toast } from 'primereact/toast';
import { Link } from 'react-router-dom';
import ModelDropdown from '@/components/ModelDropdown';

function InferencePage() {
    const imageInputRef = useRef<HTMLInputElement>(null);
    const [confidence, setConfidence] = useState<number>(0.5); // Default confidence
    const [modelName, setModelName] = useState<string>('detr_simple_demo');
    const [loading, setLoading] = useState<boolean>(false);
    const [uploadedImage, setUploadedImage] = useState<string | null>(null);
    const [annotatedImage, setAnnotatedImage] = useState<string | null>(null);
    const [metrics, setMetrics] = useState<Record<string, number> | null>(null);
    const msgsRef = useRef<Messages>(null);
    const [examples, setExamples] = useState<string[]>([]);
    const toastRef = useRef<Toast | null>(null);

    useEffect(() => {
        if (examples.length === 0) {
            inferenceExamples().then((data) => setExamples(data));
        }
    }, []);

    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        try {
            const base64Image = await convertFileToBase64(file);
            setUploadedImage(base64Image);
        } catch (error) {
            toastRef.current?.show({
                severity: 'error',
                detail: 'Failed to upload image',
            });
        }
    };

    const uploadExample = async (exampleImage: string) => {
        try {
            const response = await fetch(`${config.backend.baseUrl}/images/${exampleImage}`);
            const blob = await response.blob();
            const base64Image = await convertFileToBase64(new File([blob], exampleImage));
            setUploadedImage(base64Image);
        } catch (error) {
            toastRef.current?.show({
                severity: 'error',
                detail: 'Failed to upload example image',
            });
        }
    };

    const inferenceW = async () => {
        if (uploadedImage === null) {
            msgsRef.current?.show([{ severity: 'error', detail: 'Upload an image is required.' }]);
            return;
        }
        const request: TInferenceRequest = {
            model_name: modelName,
            confidence,
            image_base64: uploadedImage!,
        };
        setLoading(true);
        try {
            const response: TInferenceResponse = await inference(request);
            setAnnotatedImage(`data:image/jpeg;base64,${response.annotated_image_base64}`);
            setMetrics(response.metrics);
        } catch (error: any) {
            if (error.response.data.detail.export_required) {
                msgsRef.current?.show([
                    {
                        severity: 'error',
                        detail: (
                            <div className="flex flex-col">
                                <span>{`For model ${modelName} export is required.`}</span>{' '}
                                <span className="flex gap-2 items-center">
                                    Go to the export page in order to export the model
                                    <Link to="/export">
                                        <i className="pi pi-arrow-up-right" />
                                    </Link>
                                </span>
                            </div>
                        ),
                    },
                ]);
            } else {
                toastRef.current?.show({
                    severity: 'error',
                    detail: error.response.data.detail,
                });
            }
            console.error('Inference failed:', error);
            setAnnotatedImage(null);
            setMetrics(null);
        } finally {
            setLoading(false);
        }
    };

    const convertFileToBase64 = (file: File): Promise<string> => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => {
                const base64String = (reader.result as string)?.split(',')[1]; // Remove prefix
                resolve(base64String);
            };
            reader.onerror = (error) => reject(error);
        });
    };

    return (
        <div className="w-full h-full px-8 py-4 flex flex-col items-center">
            <Toast ref={toastRef} />
            <h2 className="text-center text-2xl mb-4">Model Inference</h2>

            <div className="grid grid-cols-2 gap-8 w-full max-w-[80%]">
                {/* Left Panel: Upload and Settings */}
                <div className="shadow-md rounded-md p-6 flex flex-col items-center relative">
                    {uploadedImage === null ? (
                        <>
                            <label className="flex flex-col items-center cursor-pointer w-full h-36">
                                <div className="flex flex-col items-center text-center">
                                    <i className="pi pi-cloud-upload text-6xl text-green-700"></i>
                                    <span className="mt-2 text-lg">
                                        Drag and drop an image or click to select
                                    </span>
                                </div>
                                <input
                                    type="file"
                                    ref={imageInputRef}
                                    onChange={handleFileChange}
                                    className="hidden"
                                    accept="image/*"
                                />
                            </label>
                        </>
                    ) : (
                        <>
                            <h3 className="text-lg font-semibold">Uploaded Image:</h3>
                            <Image
                                src={`data:image/jpeg;base64,${uploadedImage}`}
                                alt="Uploaded Image"
                                className="mt-4"
                            />
                            <div className="absolute top-2 right-2">
                                <Button
                                    icon="pi pi-times"
                                    className="shadow w-8 h-8 bg-red-600"
                                    tooltip="Remove Uploaded Image"
                                    severity="danger"
                                    onClick={() => {
                                        setAnnotatedImage(null);
                                        setUploadedImage(null);
                                        setLoading(false);
                                    }}
                                />
                            </div>
                        </>
                    )}

                    <div className="grid grid-cols-9 mt-4 w-full max-w-full gap-2">
                        <div className="col-span-4 flex flex-col justify-center gap-2">
                            <span className="pl-1">Model Selection:</span>
                            <ModelDropdown modelName={modelName} onChange={setModelName} />
                        </div>
                        <div className="col-span-4 flex flex-col justify-center gap-2">
                            <span className="pl-1">Confidence:</span>
                            <InputNumber
                                value={confidence}
                                onValueChange={(e) => setConfidence(e.value || 0.5)}
                                min={0}
                                max={1}
                                step={0.1}
                                showButtons
                                mode="decimal"
                                className="w-full h-10 shadow rounded"
                                pt={{
                                    input: {
                                        root: {
                                            className: 'w-[95%] pl-4',
                                        },
                                    },
                                }}
                            />
                        </div>
                        <div className="col-span-1 flex items-center justify-center">
                            <Button
                                tooltip="Run Inference"
                                icon="pi pi-bolt"
                                onClick={inferenceW}
                                className="bg-primary text-white mt-4 shadow-md p-1 min-h-12 min-w-12 max-h-12 max-w-12"
                                disabled={loading}
                            />
                        </div>
                    </div>
                    <Messages ref={msgsRef} />
                </div>

                {/* Right Panel: Results */}
                <div className="shadow-md rounded-md p-6 flex flex-col items-center justify-center">
                    {loading ? (
                        <ProgressSpinner />
                    ) : annotatedImage ? (
                        <>
                            <h3 className="text-lg font-semibold">Inference Result:</h3>
                            <Image src={annotatedImage} alt="Annotated Image" className="mt-4" />
                            <div className="mt-4 w-60">
                                <h4 className="font-semibold">Metrics:</h4>
                                <ul className="text-sm">
                                    {metrics &&
                                        Object.entries(metrics).map(([key, value]) => (
                                            <li key={key} className="flex justify-between">
                                                <span>{key}:</span>
                                                <span>{value.toFixed(2)}</span>
                                            </li>
                                        ))}
                                </ul>
                            </div>
                        </>
                    ) : (
                        <div className="p-6 max-w-md mx-auto">
                            <h2 className="text-xl font-semibold mb-4">Steps to Follow:</h2>
                            <ol className="list-decimal list-inside">
                                <li className="mb-2">Upload an image</li>
                                <li className="mb-2">Change model config (name and confidence)</li>
                                <li className="mb-2">
                                    <span>Run inference</span> <i className="pi pi-bolt" />
                                </li>
                            </ol>
                        </div>
                    )}
                </div>
            </div>

            <div className="mt-7 flex flex-col gap-2">
                <h3>Examples:</h3>
                <div className="flex gap-2">
                    {examples.map((example) => (
                        <>
                            <Tooltip target=".example-img" position="bottom" />
                            <Image
                                src={`${config.backend.baseUrl}/images/${example}`}
                                className="example-img flex w-40 cursor-pointer"
                                onClick={() => uploadExample(example)}
                                data-pr-tooltip="Upload it"
                            />
                        </>
                    ))}
                </div>
            </div>
        </div>
    );
}

export default InferencePage;
