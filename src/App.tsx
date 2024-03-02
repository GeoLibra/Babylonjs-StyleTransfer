import { useEffect, useRef } from 'react';
import '@tensorflow/tfjs-backend-webgpu';
import StyleTransferDemo from './StyleTransfer';
import './App.css';

const App = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);


  useEffect( () => {
    const demo = new StyleTransferDemo(canvasRef.current as HTMLCanvasElement);
     demo.startRenderLoop();
  }, []);
  return (
    <div>
      <canvas id="renderCanvas" width="800" height="500" ref={canvasRef}></canvas>
    </div>
  );
};

export default App;
