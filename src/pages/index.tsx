import styles from '@/styles/Home.module.css';
import { Camera } from '@mediapipe/camera_utils';
import {
  Data,
  drawConnectors,
  drawLandmarks,
  lerp,
} from '@mediapipe/drawing_utils';
import { HAND_CONNECTIONS } from '@mediapipe/hands';
import {
  FACEMESH_TESSELATION,
  Holistic,
  InputImage,
  POSE_CONNECTIONS,
  POSE_LANDMARKS_LEFT,
  POSE_LANDMARKS_RIGHT,
  Results,
  VERSION,
} from '@mediapipe/holistic';
import * as tf from '@tensorflow/tfjs';
import Head from 'next/head';
import React, { useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import { drawRect } from '../util/drawRect';

export default function Home() {
  const webcamRef = useRef<null | Webcam>(null);
  const canvasRef = useRef<null | HTMLCanvasElement>(null);
  const [showLandmarks, setShowLandmarks] = useState(false);
  const [result, setResult] = useState(null);
  let camera;

  // onResult is used only for drawing landmarks on video feed
  const onResult = (results: Results) => {
    if (canvasRef.current) {
      const canvasCtx = canvasRef.current.getContext(`2d`)!;
      canvasCtx.save();
      canvasCtx.clearRect(
        0,
        0,
        canvasRef.current.width,
        canvasRef.current.height,
      );
      canvasCtx.drawImage(
        results.image,
        0,
        0,
        canvasRef.current.width,
        canvasRef.current.height,
      );

      // Connect elbows to hands. Do this first so that the other graphics will draw
      // on top of these marks.
      canvasCtx.lineWidth = 1;

      const sequence = Array.prototype.concat(
        results.poseLandmarks
          ? results.poseLandmarks.flatMap((x) => Object.values(x))
          : new Array(132).fill(0),
        results.faceLandmarks
          ? results.faceLandmarks.flatMap((x) => Object.values(x))
          : new Array(1404).fill(0),
        results.leftHandLandmarks
          ? results.leftHandLandmarks.flatMap((x) => Object.values(x))
          : new Array(21 * 3).fill(0),
        results.rightHandLandmarks
          ? results.rightHandLandmarks.flatMap((x) => Object.values(x))
          : new Array(21 * 3).fill(0),
      );
      console.log(sequence);

      // Pose...
      drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {
        lineWidth: 0.5,
        color: `white`,
      });
      drawLandmarks(
        canvasCtx,
        Object.values(POSE_LANDMARKS_LEFT).map(
          (index) => results.poseLandmarks[index],
        ),
        {
          lineWidth: 0.5,
          visibilityMin: 0.65,
          color: `white`,
          fillColor: `rgb(255,138,0)`,
          radius: (data: Data) => {
            return lerp(data.from!.z!, -0.15, 0.1, 2, 1);
          },
        },
      );
      drawLandmarks(
        canvasCtx,
        Object.values(POSE_LANDMARKS_RIGHT).map(
          (index) => results.poseLandmarks[index],
        ),
        {
          lineWidth: 0.5,
          visibilityMin: 0.65,
          color: `white`,
          fillColor: `rgb(0,217,231)`,
          radius: (data: Data) => {
            return lerp(data.from!.z!, -0.15, 0.1, 2, 1);
          },
        },
      );

      // Hands...
      drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, {
        color: `white`,
      });
      drawLandmarks(canvasCtx, results.rightHandLandmarks, {
        color: `white`,
        fillColor: `rgb(0,217,231)`,
        lineWidth: 0.5,
        radius: (data: Data) => {
          return lerp(data.from!.z!, -0.15, 0.1, 2, 1);
        },
      });
      drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, {
        color: `white`,
      });
      drawLandmarks(canvasCtx, results.leftHandLandmarks, {
        color: `white`,
        fillColor: `rgb(255,138,0)`,
        lineWidth: 0.5,
        radius: (data: Data) => {
          return lerp(data.from!.z!, -0.15, 0.1, 2, 1);
        },
      });

      // Face...
      drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_TESSELATION, {
        color: `#C0C0C070`,
        lineWidth: 1,
        visibilityMin: 0.7,
      });
      canvasCtx.restore();
    }
  };

  const detect = async (net: tf.GraphModel) => {
    // Check data is available
    if (
      webcamRef.current &&
      webcamRef.current.video &&
      canvasRef.current &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Set canvas height and width
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      // 4. TODO - Make Detections
      const img = tf.browser.fromPixels(video);
      const resized = tf.image.resizeBilinear(img, [640, 480]);
      const casted = resized.cast(`int32`);
      const expanded = casted.expandDims(0);
      const obj = (await net.executeAsync(expanded)) as any;

      const boxes = await obj[1].array();
      const classes = await obj[2].array();
      const scores = await obj[4].array();

      setResult(boxes[0]);

      // Draw mesh
      const ctx = canvasRef.current.getContext(`2d`);

      // 5. TODO - Update drawing utility
      requestAnimationFrame(() => {
        drawRect(
          boxes[0],
          classes[0],
          scores[0],
          0.3,
          videoWidth,
          videoHeight,
          ctx,
        );
      });

      tf.dispose(img);
      tf.dispose(resized);
      tf.dispose(casted);
      tf.dispose(expanded);
      tf.dispose(obj);
    }
  };

  const runCoco = async () => {
    // const net = await tf.loadLayersModel(
    //   `https://deeplearninghackathontensorflow.s3.jp-tok.cloud-object-storage.appdomain.cloud/model.json`,
    // );
    const net = await tf.loadGraphModel(
      `https://deeplearninghackathontensorflow.s3.jp-tok.cloud-object-storage.appdomain.cloud/model.json
      `,
    );

    //  Loop and detect hands
    setInterval(() => {
      detect(net);
    }, 16.7);
  };

  useEffect(() => {
    if (showLandmarks) {
      const holistic = new Holistic({
        locateFile: (file) => {
          return (
            `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@` +
            `${VERSION}/${file}`
          );
        },
      });
      holistic.setOptions({
        smoothLandmarks: true,
        enableSegmentation: true,
        smoothSegmentation: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      holistic.onResults(onResult);
      if (webcamRef.current && webcamRef.current.video) {
        // eslint-disable-next-line react-hooks/exhaustive-deps
        camera = new Camera(webcamRef.current.video, {
          onFrame: async () => {
            await holistic.send({
              image: webcamRef.current?.video as InputImage,
            });
          },
          width: 640,
          height: 480,
        });
        camera.start();
      }
    } else {
      canvasRef.current
        ?.getContext(`2d`)
        ?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  }, [showLandmarks]);

  return (
    <div className={styles.container}>
      <Head>
        <title>MLDA Deep Learning Hackathon</title>
        <meta name="description" content="Sign Language Model" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={styles.main}>
        <h1 className={styles.title}>Welcome to openLang</h1>
        {/* <div style={{ display: `flex` }}> */}
        <Webcam
          ref={webcamRef}
          muted={true}
          className="input_video"
          style={{
            position: `absolute`,
            marginLeft: `auto`,
            marginRight: `auto`,
            left: 0,
            right: 0,
            textAlign: `center`,
            zIndex: 9,
            width: 640,
            height: 480,
          }}
        />
        <canvas
          ref={canvasRef}
          className="output_video"
          style={{
            position: `absolute`,
            marginLeft: `auto`,
            marginRight: `auto`,
            left: 0,
            right: 0,
            textAlign: `center`,
            zIndex: 8,
            width: 640,
            height: 480,
          }}
        />
        {/* </div> */}
        <span>{showLandmarks ? `true` : `false`}</span>
        <span>{JSON.stringify(result)}</span>
      </main>

      <footer className={styles.footer}>
        <button onClick={runCoco}>Detect</button>
        <button onClick={() => setShowLandmarks((state) => !state)}>
          Landmarks
        </button>
        <span>openLang</span>
      </footer>
    </div>
  );
}
