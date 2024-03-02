import * as BABYLON from 'babylonjs';
import * as tf from '@tensorflow/tfjs';
import 'babylonjs-loaders';
import '@tensorflow/tfjs-backend-webgpu';

export default class StyleTransferDemo {
  private engine: BABYLON.Engine;
  private scene: BABYLON.Scene;
  private camera: BABYLON.ArcRotateCamera;
  private quadMesh: BABYLON.Mesh;
  private renderTargetTexture: BABYLON.RenderTargetTexture;
  private styleTransferModel!: tf.GraphModel;
  private styleBottleneck!: any;
  private transformNet!: tf.GraphModel;
  private paintCanvas: HTMLCanvasElement;
  private paintCtx: CanvasRenderingContext2D;

  constructor(canvasElement: HTMLCanvasElement) {
    // BABYLON.SceneLoader.AppendAsync("/gltf/Fox.glb");
    this.paintCanvas = document.createElement('canvas');
    this.paintCtx = this.paintCanvas.getContext(
      '2d'
    ) as CanvasRenderingContext2D;
    this.engine = new BABYLON.Engine(canvasElement, true);

    this.scene = new BABYLON.Scene(this.engine);

    this.camera = new BABYLON.ArcRotateCamera(
      'Camera',
      Math.PI / 2,
      Math.PI / 2,
      2,
      new BABYLON.Vector3(0, 17, 0),
      this.scene
    );
    this.camera.attachControl(canvasElement, true);

    this.camera.setTarget(BABYLON.Vector3.Zero());
    this.camera.attachControl(canvasElement, true);

    this.quadMesh = BABYLON.Mesh.CreateSphere('sphere1', 16, 2, this.scene);

    const pbr = new BABYLON.PBRMaterial('pbr', this.scene);
    this.quadMesh.material = pbr;
    pbr.albedoColor = new BABYLON.Color3(1.0, 0.766, 0.336);
    pbr.metallic = 1.0;
    pbr.roughness = 1.0;
    pbr.reflectionTexture = BABYLON.CubeTexture.CreateFromPrefilteredData(
      '/style-img/environment.dds',
      this.scene
    );
    pbr.albedoTexture = new BABYLON.Texture(
      '/style-img/chicago.jpg',
      this.scene
    );
    this.renderTargetTexture = new BABYLON.RenderTargetTexture(
      'rtt',
      255,
      this.scene
    );
    this.renderTargetTexture.renderList.push(this.quadMesh);
    // BABYLON.SceneLoader.ImportMesh(
    //   '',
    //   '/gltf/',
    //   'Fox.glb',
    //   this.scene,
    //   (newMeshes, particleSystems, skeletons) => {
    //     this.quadMesh = newMeshes[0];
    //     this.quadMesh.position = BABYLON.Vector3.Zero();
    //     this.renderTargetTexture = new BABYLON.RenderTargetTexture(
    //       'rtt',
    //       1024,
    //       this.scene
    //     );
    //     this.renderTargetTexture.renderList.push(this.quadMesh);
    //   }
    // );
  }

  init = async () => {
    await tf.setBackend('webgpu');
    await tf.ready();
    const styleImg = await this.loadImage('/style-img/seaport.jpg');
    this.styleTransferModel = await tf.loadGraphModel(
      'model/saved_model_style_inception_js/model.json'
    );
    this.transformNet = await tf.loadGraphModel(
      'model/saved_model_transformer_js/model.json'
    );
    this.styleBottleneck = await tf.tidy(() => {
      return this.styleTransferModel.predict(
        tf.browser
          .fromPixels(styleImg)
          .toFloat()
          .div(tf.scalar(255))
          .expandDims()
      );
    });
  };

  loadImage = async (url: string): Promise<HTMLImageElement> => {
    const response = await fetch(url);
    const blob = await response.blob();
    return new Promise((resolve) => {
      const img = new Image();
      const blobUrl = URL.createObjectURL(blob);
      img.src = blobUrl;
      img.onload = function () {
        URL.revokeObjectURL(blobUrl);
        resolve(img);
      };
    });
  };

  public async applyStyleTransfer(inputTexture: BABYLON.RenderTargetTexture) {
    if (!(inputTexture && this.quadMesh)) return;
    this.engine.endFrame();
    const pixels = (await inputTexture.readPixels()) as Uint8Array;
    const width = inputTexture.getSize().width;
    const height = inputTexture.getSize().height;
    const imageData = new ImageData(
      new Uint8ClampedArray(pixels),
      width,
      height
    );

    this.paintCanvas.width = width;
    this.paintCanvas.height = height;

    this.paintCtx.putImageData(imageData, 0, 0);

    if (this.styleTransferModel) {
      const stylized = await tf.tidy(() => {
        return this.transformNet
          .predict([
            tf.browser
              .fromPixels(imageData)
              .toFloat()
              .div(tf.scalar(255))
              .expandDims(),
            this.styleBottleneck,
          ])
          .squeeze();
      });

      const stylizedImageData = await tf.browser.toPixels(stylized);
      const [height2, width2] = stylized.shape.slice(0, 2);
      const imageData2 = new ImageData(stylizedImageData, width2, height2);
      this.paintCanvas.width = width2;
      this.paintCanvas.height = height2;
      this.paintCtx.putImageData(imageData2, 0, 0);
      const dataURL = this.paintCanvas.toDataURL('image/jpeg');

      // const downloadLink = document.createElement('a');
      // downloadLink.href = dataURL;
      // downloadLink.download = 'stylized-image.png';
      // downloadLink.click();

      const newTexture = new BABYLON.Texture(
        dataURL,
        this.scene,
        true,
        false,
        BABYLON.Texture.NEAREST_SAMPLINGMODE,
        null,
        null,
        stylizedImageData,
        true
      );
      // this.quadMesh.material.albedoTexture.dispose();
      this.quadMesh.material.albedoTexture = newTexture;
      stylized.dispose();
    }

    this.engine.beginFrame();
  }

  public async startRenderLoop() {
    await this.init();

    this.engine.runRenderLoop(() => {
      this.scene.render();
    });
    setTimeout(() => {
      this.applyStyleTransfer(this.renderTargetTexture);
    }, 10000);
  }
}
