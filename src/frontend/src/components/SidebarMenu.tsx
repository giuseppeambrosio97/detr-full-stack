import { Sidebar } from 'primereact/sidebar';
import { Menu } from 'primereact/menu';
import { useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { Button } from 'primereact/button';

export default function SidebarMenu() {
    const [visible, setVisible] = useState(false);
    const navigate = useNavigate();

    const menuItems = [
        {
            label: 'DETR',
            items: [
                {
                    label: 'DETR ARK',
                    icon: 'pi pi-box',
                    command: () => {
                        navigate('/home');
                    },
                },
                {
                    label: 'Model Inference',
                    icon: 'pi pi-bolt',
                    command: () => {
                        navigate('/inference');
                    },
                },
                {
                    label: 'Model Export ONNX',
                    icon: 'pi pi-file-export',
                    command: () => {
                        navigate('/export');
                    },
                },
            ],
        },
    ];

    return (
        <>
            <Sidebar visible={visible} onHide={() => setVisible(false)}>
                <div className="flex flex-col gap-3">
                    <h2 className="text-center font-bold">Menu</h2>
                    <Menu model={menuItems} className="w-full min-w-full" />
                </div>
            </Sidebar>

            {/* Button to toggle the sidebar */}
            <Button
                icon="pi pi-bars"
                className="shadow rounded-sm"
                size="large"
                onClick={() => setVisible(true)}
            />
        </>
    );
}
