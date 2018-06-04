using System.Collections;
using System.Collections.Generic;
using UnityEngine;
namespace RosSharp.RosBridgeClient {
    public class BallCollision : MonoBehaviour
    {

        public GameObject rosConnector;
        private RewardSender rewarder;
        private bool collided = false;

        // Use this for initialization
        void Start()
        {
            rosConnector = FindObjectOfType<RosConnector>().gameObject;
            rewarder = rosConnector.GetComponent<RewardSender>();
        }

        // Update is called once per frame
        void Update()
        {

        }

        private void OnCollisionEnter(Collision collision)
        {
            if (collision.other.tag == "Player" && collided == false)
            {
                Debug.Log("Ball as been hit by robot");
                rewarder.SendReward(10);
                collided = true;
            }
        }
    }
}
