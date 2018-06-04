using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace RosSharp.RosBridgeClient
{
    [RequireComponent(typeof(RosConnector))]
    public class RewardSender : MonoBehaviour
    {
        private RosSocket rosSocket;
        private int pub_id;

        private StandardString msg = new StandardString();

        // Use this for initialization
        void Start()
        {
            rosSocket = transform.GetComponent<RosConnector>().RosSocket;
            pub_id = rosSocket.Advertize("reward", "std_msgs/String");

        }

        public void SendReward (float r)
        {
            msg.data = r.ToString();
            rosSocket.Publish(pub_id, msg);
        }

    }
}
