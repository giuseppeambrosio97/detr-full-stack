import { Image } from "primereact/image";

export default function HomePage() {
  return (
    <div className="w-full h-full overflow-y-auto scrollbar max-h-[80%]">
      <h2 className="text-center text-2xl">DETR ARK</h2>
      <div className="flex items-center justify-center mt-10">
        <Image src="detr_ark.png"/>
      </div>
    </div>
  );
}
