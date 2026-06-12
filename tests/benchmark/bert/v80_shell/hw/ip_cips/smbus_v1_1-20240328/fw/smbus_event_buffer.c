/**
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * This file contains the function definitions for the event buffer intialize, read and write functions
 *
 * @file smbus_event_buffer.c
 *
 */

#include "smbus_internal.h"
#include "smbus_event_buffer.h"

/********************** Static function declarations ***************************/

/******************************************************************************
*
* @brief    This function increments the event buffer index until it hits the
*           max value at which point it sets it to 0.
*
* @param    pxContext is a pointer to the event buffer structure.
* @param    ulAnyIndex is the current index position
*
* @return   uint32_t new index position
*
* @note     None.
*
*****************************************************************************/
static uint32_t ulEventBufferInc( SMBUS_EVENT_BUFFER_TYPE* pxContext, uint32_t ulAnyIndex );

/*******************************************************************************/

/******************************************************************************
*
* @brief    Walks through all elements of the event buffer. It sets all the elements to unoccupied
*           and sets the event value to 0.
*
*****************************************************************************/
void vEventBufferInitialize( SMBUS_EVENT_BUFFER_TYPE* pxContext, SMBUS_EVENT_BUFFER_ELEMENT_TYPE* pxEventBuffer, uint32_t ulMaxElements )
{
    uint32_t i = 0;

    if( ( NULL != pxContext ) &&
        ( NULL != pxEventBuffer ) )
    {
        pxContext->ulMaxElements = ulMaxElements;
        pxContext->pxEventBuffer = pxEventBuffer;

        for( i = 0; i < ( pxContext->ulMaxElements ); i++ )
        {
            pxContext->pxEventBuffer[i].ucIsOccupied = SMBUS_FALSE;
            pxContext->pxEventBuffer[i].ucOctet = 0x00;
        }
        
        pxContext->ulWrite = 0;
        pxContext->ulRead = 0;
    }
}

/******************************************************************************
*
* @brief    This function increments the event buffer index until it hits the
*           max value at which point it sets it to 0.
*
*****************************************************************************/
static uint32_t ulEventBufferInc( SMBUS_EVENT_BUFFER_TYPE* pxContext, uint32_t ulAnyIndex )
{
    uint32_t ulResult = ulAnyIndex;

    if( NULL != pxContext )
    {
        ulResult++;

        if( ( pxContext->ulMaxElements ) <= ulResult )
        {
            ulResult = 0;
        }
    }
    return ( ulResult );
}

/******************************************************************************
*
* @brief    If the write location is not occupied the event is written to that
*           location. The location is marked as occupied and the write location
*           is incremented
*
*****************************************************************************/
uint8_t ucEventBufferTryWrite( SMBUS_EVENT_BUFFER_TYPE* pxContext, uint8_t ucAnyCharacter, uint32_t* pulWrite_Position )
{
    uint8_t ucResult = SMBUS_EVENT_BUFFER_FAIL;

    if( ( NULL != pxContext ) &&  
        ( NULL != pulWrite_Position ) )
    {
        if( SMBUS_FALSE == pxContext->pxEventBuffer[pxContext->ulWrite].ucIsOccupied )
        {
            *pulWrite_Position = pxContext->ulWrite;
            pxContext->pxEventBuffer[pxContext->ulWrite].ucOctet = ucAnyCharacter;
            pxContext->pxEventBuffer[pxContext->ulWrite].ucIsOccupied = SMBUS_TRUE;
            pxContext->ulWrite = ulEventBufferInc( pxContext, pxContext->ulWrite );
            ucResult = SMBUS_EVENT_BUFFER_SUCCESS;
        }
    }
    return ( ucResult );
}

/******************************************************************************
*
* @brief    If the read location is occupied the event at the location is read
*           The read location is then marked as empty and the read location incremented
*
*****************************************************************************/
uint8_t ucEventBufferTryRead( SMBUS_EVENT_BUFFER_TYPE* pxContext, uint8_t* pucAnyCharacter, uint32_t* pulRead_Position )
{
    uint8_t ucResult = SMBUS_FALSE;

    if( ( NULL != pxContext ) &&
        ( NULL != pucAnyCharacter ) &&
        ( NULL != pulRead_Position ) )
    {
        *pucAnyCharacter = 0xCC;
        
        if( SMBUS_TRUE == pxContext->pxEventBuffer[pxContext->ulRead].ucIsOccupied )
        {
            *pulRead_Position = pxContext->ulRead;
            *pucAnyCharacter = pxContext->pxEventBuffer[pxContext->ulRead].ucOctet;
            pxContext->pxEventBuffer[pxContext->ulRead].ucIsOccupied = SMBUS_FALSE;
            pxContext->ulRead = ulEventBufferInc( pxContext, pxContext->ulRead );
            ucResult = SMBUS_TRUE;
        }
    }
    
    return ( ucResult );
}
