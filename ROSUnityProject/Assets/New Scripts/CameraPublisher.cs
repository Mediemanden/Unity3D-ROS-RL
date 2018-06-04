
using UnityEngine;

namespace RosSharp.RosBridgeClient
{

    [RequireComponent(typeof(RosConnector))]
    public class CameraPublisher : MonoBehaviour 
    {
        public Camera cam;
        private RosSocket rosSocket;
        public int UpdateTime = 1;
        private Texture2D viewData;
        private SensorCompressedImage img = new SensorCompressedImage();
        private float time;
        private int pub_id;
        private VelocitySubscriber velSub;
        public GameObject ball;
        private float prevDistance;
        public GameObject turtle;
        

        public void Start()
        {
            ball = GameObject.FindGameObjectWithTag("Ball");
            velSub = gameObject.GetComponent<VelocitySubscriber>();
            velSub.actionReceived = true;
            viewData = new Texture2D(cam.targetTexture.width, cam.targetTexture.height);
            rosSocket = transform.GetComponent<RosConnector>().RosSocket;
            img.format = "png";
            time = Time.time;
            cam.depthTextureMode = DepthTextureMode.Depth;
            pub_id = rosSocket.Advertize("camera/image", "sensor_msgs/CompressedImage");
        }
        public void Update()
        {

            if (velSub.actionReceived == true)
            {
                Vector3 screenPoint = cam.WorldToViewportPoint(ball.transform.position);
                bool onScreen = screenPoint.z > 0 && screenPoint.x > 0 && screenPoint.x < 1 && screenPoint.y > 0 && screenPoint.y < 1;
                float currentDistance = Vector3.Distance(ball.transform.position, turtle.transform.position);
                
                if (onScreen == true && (prevDistance > (currentDistance + 0.01)))
                {
                    gameObject.GetComponent<RewardSender>().SendReward(-0.1f);
                }
                else if (onScreen == true && (prevDistance < (currentDistance - 0.01)))
                {
                    gameObject.GetComponent<RewardSender>().SendReward(-0.2f);
                }
                else
                {
                    gameObject.GetComponent<RewardSender>().SendReward(-0.4f);
                }
                prevDistance = currentDistance;
                velSub.actionReceived = false;
                time = Time.time;
            }

            UpdateFeed();
            rosSocket.Publish(pub_id, img);
                
               
              
        }

        private void UpdateFeed()
        {

            RenderTexture.active = cam.targetTexture;
            viewData.ReadPixels(new Rect(0, 0, cam.targetTexture.width, cam.targetTexture.height), 0, 0);
            viewData.Apply();

            img.data = ImageConversion.EncodeToPNG(viewData);

        }
    }
}