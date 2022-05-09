using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Animate : MonoBehaviour
{
    private int currentIndex = 0;
    Material vertexColorShaderMaterial;
    private GameObject rootObject;
    private int fps = 0;
    
    private void Awake()
    {
        
        Debug.Log("Waking up");
        rootObject = GameObject.Find("Animator");
        vertexColorShaderMaterial = Resources.Load("VertexColorMaterial", typeof(Material)) as Material;
        Debug.Log(string.Format("Total Frames {0}", rootObject.transform.childCount));
    }

    private void Update()
    {
        fps++;
        if(fps % 10 == 0)
        {
            if(currentIndex == rootObject.transform.childCount-1)
            {
                #if UNITY_EDITOR
                    // Application.Quit() does not work in the editor so
                    // UnityEditor.EditorApplication.isPlaying need to be set to false to end the game
                    UnityEditor.EditorApplication.isPlaying = false;
                #else
                    Application.Quit();
                #endif
            }

            Debug.Log(currentIndex);

            if(currentIndex > 0)
            {
                Debug.Log("Removing older objects");
                rootObject.transform.GetChild(currentIndex-1).gameObject.SetActive(false);
            }

            

            GameObject frame = rootObject.transform.GetChild(currentIndex).gameObject;
            frame.GetComponent<Renderer>().material = vertexColorShaderMaterial;
            frame.SetActive(true);

            currentIndex = (currentIndex + 1);
        }

        
        
    }
}