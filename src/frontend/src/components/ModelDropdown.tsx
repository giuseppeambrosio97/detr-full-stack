import { Dropdown } from 'primereact/dropdown';

type ModelDropdownProps = {
    modelName: string;
    onChange: (newModelName: string) => void;
};

const MODEL_NAMES = ['detr_simple_demo', 'detr_resnet101_panoptic', 'detr_simple_demo_onnx'];

export default function ModelDropdown(props: ModelDropdownProps) {
    return (
        <Dropdown
            value={props.modelName}
            onChange={(e) => props.onChange(e.target.value)}
            options={MODEL_NAMES}
            className="h-10 p-2 w-full shadow"
            pt={{
                input: {
                    className: 'p-2 m-0 flex items-center justify-center',
                },
            }}
        />
    );
}
