using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BallSpawner : MonoBehaviour {

    public GameObject ball;
	// Use this for initialization
	void Awake () {

        if( Random.Range(0.0f,1.0f) >= 0.5)
        {
            Vector3 pos = new Vector3(Random.Range(-1.0f, -2.2f), 0.1f, Random.Range(-1.1f, 1.1f));
            Instantiate(ball, pos, Quaternion.identity);
        }
        else
        {
            Vector3 pos = new Vector3(Random.Range(1.0f, 2.2f), 0.1f, Random.Range(-1.1f, 1.1f));
            Instantiate(ball, pos, Quaternion.identity);
        }
        
        

	}
	
	// Update is called once per frame
	void Update () {
		
	}
}
