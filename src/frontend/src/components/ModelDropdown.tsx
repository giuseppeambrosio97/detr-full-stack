import { modelList, ModelType } from '@/api/model-list';
import { Dropdown } from 'primereact/dropdown';
import { useEffect, useState } from 'react';

type ModelDropdownProps = {
    modelName: string;
    onChange: (newModelName: string) => void;
    modelType: ModelType;
};

export default function ModelDropdown(props: ModelDropdownProps) {
    const [models, setModels] = useState<string[]>([]);

    useEffect(() => {
        if(models.length === 0) {
            modelList(props.modelType)
            .then(data => setModels(data));
        }
    }, []);

    return (
        <Dropdown
            value={props.modelName}
            onChange={(e) => props.onChange(e.target.value)}
            options={models}
            className="h-10 p-2 w-full shadow"
            pt={{
                input: {
                    className: 'p-2 m-0 flex items-center justify-center',
                },
            }}
        />
    );
}
